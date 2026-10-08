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
  /** Packed two-operand op: input, len, a_len, out buffer, out length. */
  const packedPair = (name) =>
    lib.func(name, "int32_t", [
      "const double *",
      "size_t",
      "size_t",
      koffi.out(koffi.pointer("void *")),
      koffi.out(koffi.pointer("size_t")),
    ]);
  /** complex_powi: input, len, n, out buffer, out length. */
  const powi = (name) =>
    lib.func(name, "int32_t", [
      "const double *",
      "size_t",
      "size_t",
      koffi.out(koffi.pointer("void *")),
      koffi.out(koffi.pointer("size_t")),
    ]);
  /** RANSAC: input, len, threshold, iterations, seed, out buffer, out length. */
  const ransac = (name) =>
    lib.func(name, "int32_t", [
      "const double *",
      "size_t",
      "double",
      "size_t",
      "uint64_t",
      koffi.out(koffi.pointer("void *")),
      koffi.out(koffi.pointer("size_t")),
    ]);
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
    complex_mul: alloc("pith_math_complex_mul"),
    complex_div: alloc("pith_math_complex_div"),
    complex_exp: alloc("pith_math_complex_exp"),
    complex_log: alloc("pith_math_complex_log"),
    complex_sqrt: alloc("pith_math_complex_sqrt"),
    complex_powi: powi("pith_math_complex_powi"),
    complex_arg: scalar("pith_math_complex_arg"),
    fft_n: alloc("pith_math_fft_n"),
    ifft_n: alloc("pith_math_ifft_n"),
    dct3: alloc("pith_math_dct3"),
    dct3_2d: alloc2d("pith_math_dct3_2d"),
    conv: packedPair("pith_math_conv"),
    corr: packedPair("pith_math_corr"),
    mean: scalar("pith_math_mean"),
    variance: scalar("pith_math_var"),
    cov: scalar("pith_math_cov"),
    lagrange: lib.func("pith_math_lagrange", "int32_t", [
      "const double *", "size_t", "double", koffi.out(koffi.pointer("double")),
    ]),
    ransac_line: ransac("pith_math_ransac_line"),
    dtw: lib.func("pith_math_dtw", "int32_t", [
      "const double *", "size_t", "size_t", koffi.out(koffi.pointer("double")),
    ]),
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

// ---------------------------------------------------------------------
// Tier-1 expansion
// ---------------------------------------------------------------------

/**
 * Complex product of two interleaved pairs `[re, im]`.
 * @param {number[]} z
 * @param {number[]} w
 * @returns {number[]}
 */
function complex_mul(z, w) {
  return callAlloc(loadLibrary().complex_mul, "pith_math_complex_mul", pack([...z, ...w]));
}

/**
 * Complex quotient `z / w`; a zero denominator throws `FfiError`
 * with `STATUS_REJECTED`.
 * @param {number[]} z
 * @param {number[]} w
 * @returns {number[]}
 */
function complex_div(z, w) {
  return callAlloc(loadLibrary().complex_div, "pith_math_complex_div", pack([...z, ...w]));
}

/** `e^z` for the interleaved pair `z`. */
function complex_exp(z) {
  return callAlloc(loadLibrary().complex_exp, "pith_math_complex_exp", pack(z));
}

/** Principal `[ln|z|, arg z]`; `z = 0` throws with `STATUS_REJECTED`. */
function complex_log(z) {
  return callAlloc(loadLibrary().complex_log, "pith_math_complex_log", pack(z));
}

/** Principal square root of the interleaved pair `z`. */
function complex_sqrt(z) {
  return callAlloc(loadLibrary().complex_sqrt, "pith_math_complex_sqrt", pack(z));
}

/**
 * Integer power `z^n` by squaring (non-negative `n`).
 * @param {number[]} z
 * @param {number} n
 * @returns {number[]}
 */
function complex_powi(z, n) {
  const fn = loadLibrary().complex_powi;
  const buf = pack(z);
  const out = [null];
  const outLen = [0];
  const status = fn(buf, buf === null ? 0 : buf.length, n, out, outLen);
  if (status !== STATUS_OK) throw new FfiError("pith_math_complex_powi", status);
  try {
    return [...new Float64Array(koffi.decode(out[0], koffi.array("double", outLen[0] / 8)))];
  } finally {
    loadLibrary().free(out[0], outLen[0]);
  }
}

/** Principal argument of `z` in radians, `(-π, π]`. */
function complex_arg(z) {
  return callScalar(loadLibrary().complex_arg, "pith_math_complex_arg", pack(z));
}

/** Forward DFT of **any** `n ≥ 1` (Bluestein); interleaved pairs. */
function fft_n(x) {
  return callAlloc(loadLibrary().fft_n, "pith_math_fft_n", pack(x));
}

/** Inverse of {@link fft_n}. */
function ifft_n(x) {
  return callAlloc(loadLibrary().ifft_n, "pith_math_ifft_n", pack(x));
}

/** Orthonormal DCT-III — the exact inverse of {@link dct2}. */
function dct3(x) {
  return callAlloc(loadLibrary().dct3, "pith_math_dct3", pack(x));
}

/** Separable 2D orthonormal DCT-III; the inverse of {@link dct2_2d}. */
function dct3_2d(x, w, h) {
  return callAlloc(loadLibrary().dct3_2d, "pith_math_dct3_2d", pack(x), w, h);
}

/**
 * Full-support linear convolution — `x.length + k.length − 1` samples.
 * @param {number[]} x
 * @param {number[]} k
 * @returns {number[]}
 */
function conv(x, k) {
  return callAlloc(loadLibrary().conv, "pith_math_conv", pack([...x, ...k]), x.length);
}

/** Cross-correlation — {@link conv} with the flipped kernel. */
function corr(x, k) {
  return callAlloc(loadLibrary().corr, "pith_math_corr", pack([...x, ...k]), x.length);
}

/** Arithmetic mean; empty input throws `STATUS_INVALID`. */
function mean(x) {
  return callScalar(loadLibrary().mean, "pith_math_mean", pack(x));
}

/** Sample variance (`n − 1` denominator); fewer than two throws. */
function variance(x) {
  return callScalar(loadLibrary().variance, "pith_math_var", pack(x));
}

/** Sample covariance of interleaved pairs; short input throws. */
function cov(x) {
  return callScalar(loadLibrary().cov, "pith_math_cov", pack(x));
}

/**
 * Lagrange interpolation through `(xs[i], ys[i])` at `x`.
 * @param {number[]} xs
 * @param {number[]} ys
 * @param {number} x
 * @returns {number}
 */
function lagrange(xs, ys, x) {
  if (xs.length !== ys.length) {
    throw new FfiError("pith_math_lagrange", STATUS_INVALID);
  }
  const packed = [];
  for (let i = 0; i < xs.length; i++) packed.push(xs[i], ys[i]);
  const fn = loadLibrary().lagrange;
  const buf = pack(packed);
  const out = [0];
  const status = fn(buf, buf === null ? 0 : buf.length, x, out);
  if (status !== STATUS_OK) throw new FfiError("pith_math_lagrange", status);
  return out[0];
}

/**
 * Seeded RANSAC line fit: `[slope, intercept, inliers]`, or `null`
 * when no model was found.
 * @param {number[]} xs
 * @param {number[]} ys
 * @param {number} threshold
 * @param {number} iterations
 * @param {number} seed
 * @returns {[number, number, number]|null}
 */
function ransac_line(xs, ys, threshold, iterations, seed) {
  if (xs.length !== ys.length) {
    throw new FfiError("pith_math_ransac_line", STATUS_INVALID);
  }
  const packed = [];
  for (let i = 0; i < xs.length; i++) packed.push(xs[i], ys[i]);
  const fn = loadLibrary().ransac_line;
  const buf = pack(packed);
  const out = [null];
  const outLen = [0];
  const status = fn(buf, buf === null ? 0 : buf.length, threshold, iterations, seed, out, outLen);
  if (status === STATUS_REJECTED) return null;
  if (status !== STATUS_OK) throw new FfiError("pith_math_ransac_line", status);
  try {
    const vals = [...new Float64Array(koffi.decode(out[0], koffi.array("double", outLen[0] / 8)))];
    return [vals[0], vals[1], vals[2]];
  } finally {
    loadLibrary().free(out[0], outLen[0]);
  }
}

/**
 * Dynamic time warping distance (absolute-difference local cost,
 * three monotone steps, optimal path); empty input throws
 * `STATUS_INVALID`.
 * @param {number[]} a
 * @param {number[]} b
 * @returns {number}
 */
function dtw(a, b) {
  const fn = loadLibrary().dtw;
  const buf = pack([...a, ...b]);
  const out = [0];
  const status = fn(buf, buf === null ? 0 : buf.length, a.length, out);
  if (status !== STATUS_OK) throw new FfiError("pith_math_dtw", status);
  return out[0];
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
  complex_mul,
  complex_div,
  complex_exp,
  complex_log,
  complex_sqrt,
  complex_powi,
  complex_arg,
  fft_n,
  ifft_n,
  dct3,
  dct3_2d,
  conv,
  corr,
  mean,
  variance,
  cov,
  lagrange,
  ransac_line,
  dtw,
};
