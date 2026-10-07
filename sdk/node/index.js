// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash
"use strict";

/**
 * pith-math SDK: FFT, DCT-II, median and 3×3 linear algebra through koffi.
 *
 * The single Rust core (the `pith-math` cdylib built by
 * `cargo build --release`) is loaded at runtime; `koffi` is the only
 * runtime dependency.
 *
 * Discovery order (the suite's cdylib convention):
 *
 * 1. `PITH_CDYLIB` — an explicit cdylib *file* path;
 * 2. `PITH_CDYLIB_DIR` — a *directory* scanned for the cdylib names
 *    (the CD pipeline points this at `target/release`);
 * 3. `prebuilds/<platform>-<arch>/` then `prebuilds/` flat (the CD
 *    publish copies the built cdylib there);
 * 4. `<repo root>/target/release` — the repository working-tree layout,
 *    so a source checkout runs against a local cargo build unconfigured.
 *
 * Every transform takes and returns plain `number[]` buffers (floats);
 * the scalar order statistics (`median`, `det3`) return a number.
 * Non-zero FFI statuses throw `FfiError` carrying the raw status code —
 * the cdylib never panics through this boundary.
 */

const koffi = require("koffi");
const fs = require("node:fs");
const path = require("node:path");

const STATUS_OK = 0;
const STATUS_INVALID = -1;
const STATUS_REJECTED = -2;

/** Every cdylib file name cargo may drop into the build directory, per platform. */
const CDYLIB_NAMES = ["pith_math.dll", "libpith_math.so", "libpith_math.dylib"];

const PKG_ROOT = path.join(__dirname);
const REPO_ROOT = path.resolve(__dirname, "..", "..");

/** FfiError: a non-zero status code came back from the cdylib. */
class FfiError extends Error {
  /**
   * @param {string} op the FFI operation name
   * @param {number} status the raw status code
   */
  constructor(op, status) {
    const kind = { [STATUS_INVALID]: "invalid argument", [STATUS_REJECTED]: "input rejected" }[status] ?? "unknown failure";
    super(`${op} failed: ${kind} (status ${status})`);
    this.name = "FfiError";
    /** The raw status code the FFI returned. */
    this.status = status;
  }
}

/**
 * Locates the cdylib through the suite's discovery chain.
 * @returns {string} an absolute path to the cdylib file
 * @throws {Error} when nothing is found
 */
function findCdylib() {
  const explicit = process.env.PITH_CDYLIB;
  if (explicit && fs.statSync(explicit, { throwIfNoEntry: false })?.isFile()) {
    return path.resolve(explicit);
  }
  /** @type {string[]} */
  const dirs = [];
  const envDir = process.env.PITH_CDYLIB_DIR;
  if (envDir) {
    dirs.push(envDir);
    if (!path.isAbsolute(envDir)) {
      dirs.push(path.join(REPO_ROOT, envDir));
    }
  }
  const osArch = `${process.platform}-${process.arch}`;
  dirs.push(path.join(PKG_ROOT, "prebuilds", osArch));
  dirs.push(path.join(PKG_ROOT, "prebuilds"));
  dirs.push(path.join(REPO_ROOT, "target", "release"));
  for (const dir of dirs) {
    for (const name of CDYLIB_NAMES) {
      const p = path.join(dir, name);
      if (fs.statSync(p, { throwIfNoEntry: false })?.isFile()) return p;
    }
  }
  throw new Error(
    "no pith-math cdylib found (searched PITH_CDYLIB, PITH_CDYLIB_DIR, prebuilds/ and <repo>/target/release); " +
      "run `cargo build --release` first",
  );
}

let cached = undefined;

/**
 * Loads the cdylib and binds the exported symbols (lazily, once).
 * @returns {Record<string, Function>}
 */
function loadLibrary() {
  if (cached) return cached;
  const lib = koffi.load(findCdylib());
  /** Flat-array op: input, len, out buffer, out length (bytes). */
  const alloc = (name) =>
    lib.func(name, "int32_t", [
      "const double *",
      "size_t",
      koffi.out(koffi.pointer("void *")),
      koffi.out(koffi.pointer("size_t")),
    ]);
  /** 2D op: input, len, w, h, out buffer, out length (bytes). */
  const alloc2d = (name) =>
    lib.func(name, "int32_t", [
      "const double *",
      "size_t",
      "size_t",
      "size_t",
      koffi.out(koffi.pointer("void *")),
      koffi.out(koffi.pointer("size_t")),
    ]);
  /** Scalar-out op: input, len, out slot. */
  const scalar = (name) =>
    lib.func(name, "int32_t", ["const double *", "size_t", koffi.out(koffi.pointer("double"))]);
  cached = {
    dct2: alloc("pith_math_dct2"),
    idct2: alloc("pith_math_idct2"),
    dct2_2d: alloc2d("pith_math_dct2_2d"),
    fft: alloc("pith_math_fft"),
    ifft: alloc("pith_math_ifft"),
    fft_real: alloc("pith_math_fft_real"),
    solve3: alloc("pith_math_solve3"),
    det3: scalar("pith_math_det3"),
    inverse3: alloc("pith_math_inverse3"),
    transpose3: alloc("pith_math_transpose3"),
    mat3_mul: alloc("pith_math_mat3_mul"),
    mat3_mul_vec: alloc("pith_math_mat3_mul_vec"),
    median: scalar("pith_math_median"),
    free: lib.func("void pith_math_free(void *ptr, size_t len)"),
  };
  return cached;
}

/**
 * Packs the caller's array (or `null`, for the NULL-pointer refusal
 * path) into a Float64Array koffi can hand to `const double *`.
 * @param {number[]|Float64Array|null} x
 * @returns {Float64Array|null}
 */
function pack(x) {
  if (x === null || x === undefined) return null;
  return x instanceof Float64Array ? x : Float64Array.from(x);
}

/**
 * Runs one allocating op and copies the handed-out buffer into a plain
 * array (element count = byte count / 8) before releasing it.
 * @param {Function} fn the bound koffi function
 * @param {string} op the FFI operation name, for error reporting
 * @param {Float64Array|null} input packed input
 * @param {...number} extra extra usize arguments (w, h for the 2D op)
 * @returns {number[]}
 */
function callAlloc(fn, op, input, ...extra) {
  const { free } = loadLibrary();
  const out = [null];
  const outLen = [0];
  const status = fn(input, input === null ? 0 : input.length, ...extra, out, outLen);
  if (status !== STATUS_OK) {
    throw new FfiError(op, status);
  }
  try {
    const count = Number(outLen[0]) / 8;
    // koffi.decode hands back a typed view over the external buffer;
    // copy it into an owned array before the cdylib buffer is freed.
    return Array.from(koffi.decode(out[0], "double", count));
  } finally {
    free(out[0], Number(outLen[0]));
  }
}

/**
 * Runs one scalar-out op.
 * @param {Function} fn the bound koffi function
 * @param {string} op the FFI operation name, for error reporting
 * @param {Float64Array|null} input packed input
 * @returns {number}
 */
function callScalar(fn, op, input) {
  const out = [null];
  const status = fn(input, input === null ? 0 : input.length, out);
  if (status !== STATUS_OK) {
    throw new FfiError(op, status);
  }
  return out[0];
}

/**
 * 1D orthonormal DCT-II. An empty input throws `FfiError` with
 * `status === -1`.
 * @param {number[]} x
 * @returns {number[]}
 */
function dct2(x) {
  return callAlloc(loadLibrary().dct2, "pith_math_dct2", pack(x));
}

/**
 * 1D orthonormal DCT-III — the exact inverse of {@link dct2}.
 * @param {number[]} x
 * @returns {number[]}
 */
function idct2(x) {
  return callAlloc(loadLibrary().idct2, "pith_math_idct2", pack(x));
}

/**
 * Separable 2D orthonormal DCT-II over a `w × h` row-major matrix.
 * @param {number[]} x flat matrix, `x.length` must equal `w * h`
 * @param {number} w width
 * @param {number} h height
 * @returns {number[]}
 */
function dct2_2d(x, w, h) {
  return callAlloc(loadLibrary().dct2_2d, "pith_math_dct2_2d", pack(x), w, h);
}

/**
 * Forward DFT over interleaved complex pairs `[re, im, …]`; the
 * complex count must be a non-zero power of two.
 * @param {number[]} x
 * @returns {number[]}
 */
function fft(x) {
  return callAlloc(loadLibrary().fft, "pith_math_fft", pack(x));
}

/**
 * Inverse DFT, the exact (within ulps) inverse of {@link fft}.
 * @param {number[]} x
 * @returns {number[]}
 */
function ifft(x) {
  return callAlloc(loadLibrary().ifft, "pith_math_ifft", pack(x));
}

/**
 * Forward DFT of a real signal; returns the full `2·x.length`
 * interleaved spectrum. `x.length` must be a non-zero power of two.
 * @param {number[]} x
 * @returns {number[]}
 */
function fft_real(x) {
  return callAlloc(loadLibrary().fft_real, "pith_math_fft_real", pack(x));
}

/**
 * Solves the 3×3 system `a·x = b` (row-major `a`). A singular system
 * throws `FfiError` with `status === -2`.
 * @param {number[]} a nine row-major matrix entries
 * @param {number[]} b three right-hand-side entries
 * @returns {number[]}
 */
function solve3(a, b) {
  return callAlloc(loadLibrary().solve3, "pith_math_solve3", pack([...a, ...b]));
}

/**
 * Determinant of the row-major 3×3 matrix `m`.
 * @param {number[]} m
 * @returns {number}
 */
function det3(m) {
  return callScalar(loadLibrary().det3, "pith_math_det3", pack(m));
}

/**
 * Inverse of the row-major 3×3 matrix `m`; singular input throws
 * `FfiError` with `status === -2`.
 * @param {number[]} m
 * @returns {number[]}
 */
function inverse3(m) {
  return callAlloc(loadLibrary().inverse3, "pith_math_inverse3", pack(m));
}

/**
 * Transpose of the row-major 3×3 matrix `m`.
 * @param {number[]} m
 * @returns {number[]}
 */
function transpose3(m) {
  return callAlloc(loadLibrary().transpose3, "pith_math_transpose3", pack(m));
}

/**
 * Matrix product `a·b` of two row-major 3×3 factors.
 * @param {number[]} a
 * @param {number[]} b
 * @returns {number[]}
 */
function mat3_mul(a, b) {
  return callAlloc(loadLibrary().mat3_mul, "pith_math_mat3_mul", pack([...a, ...b]));
}

/**
 * Matrix-vector product `m·v`.
 * @param {number[]} m nine row-major matrix entries
 * @param {number[]} v three vector entries
 * @returns {number[]}
 */
function mat3_mul_vec(m, v) {
  return callAlloc(loadLibrary().mat3_mul_vec, "pith_math_mat3_mul_vec", pack([...m, ...v]));
}

/**
 * The median of `x` — the lower middle element for even `n`, never the
 * mean of the two middles. Empty input throws `FfiError` with
 * `status === -1`.
 * @param {number[]} x
 * @returns {number}
 */
function median(x) {
  return callScalar(loadLibrary().median, "pith_math_median", pack(x));
}

module.exports = {
  STATUS_OK,
  STATUS_INVALID,
  STATUS_REJECTED,
  CDYLIB_NAMES,
  FfiError,
  findCdylib,
  dct2,
  idct2,
  dct2_2d,
  fft,
  ifft,
  fft_real,
  solve3,
  det3,
  inverse3,
  transpose3,
  mat3_mul,
  mat3_mul_vec,
  median,
};
