// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash
"use strict";

// Hex-exact conformance: the committed reference vectors through koffi.
// Every vector in tests/reference.json is replayed through the cdylib
// and compared per that file's own policy — bit-for-bit for `exact`
// vectors (raw IEEE-754 patterns, NaN and -0.0 safe), within
// max(tol_abs, tol_rel·|expected|) for the twiddle-factor vectors. The
// same vectors the Rust gen-reference verify gate and the Python/Go
// SDKs check.

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const {
  FfiError,
  dct2,
  dct2_2d,
  det3,
  fft,
  fft_real,
  findCdylib,
  idct2,
  ifft,
  inverse3,
  mat3_mul,
  mat3_mul_vec,
  median,
  solve3,
  transpose3,
} = require("../index.js");

const REPO_ROOT = path.resolve(__dirname, "..", "..", "..");

const REFERENCE = JSON.parse(fs.readFileSync(path.join(REPO_ROOT, "tests", "reference.json"), "utf8")).vectors;

/** 16-digit hex of the raw IEEE-754 pattern → f64. */
function bitsToFloat(h) {
  const bits = BigInt("0x" + h);
  return new Float64Array(new BigUint64Array([bits]).buffer)[0];
}

/** f64 → the raw IEEE-754 pattern as a 16-digit lowercase hex string. */
function floatToBits(v) {
  const bits = new BigUint64Array(new Float64Array([v]).buffer)[0];
  return bits.toString(16).padStart(16, "0");
}

/** Replays the vector's op composition through the SDK. */
function runOp(name, x) {
  if (name === "dct2.n8" || name === "dct2.n32") return dct2(x);
  if (name === "idct2.roundtrip.n8") return idct2(dct2(x));
  if (name === "dct2.2d.8x8" || name === "dct2.dc.8x8") return dct2_2d(x, 8, 8);
  if (name === "fft.n8" || name === "fft.n64") return fft(x);
  if (name === "ifft.roundtrip.n64") return ifft(fft(x));
  if (name === "fft.real.n8") return fft_real(x);
  if (name.startsWith("median.")) return [median(x)];
  if (name.startsWith("solve3.")) return solve3(x.slice(0, 9), x.slice(9, 12));
  if (name === "det3.standard") return [det3(x)];
  if (name === "inverse3.sym") return inverse3(x);
  if (name === "mat3.mul") return mat3_mul(x.slice(0, 9), x.slice(9, 18));
  if (name === "mat3.mul_vec.rot90") return mat3_mul_vec(x.slice(0, 9), x.slice(9, 12));
  if (name === "transpose3") return transpose3(x);
  throw new Error(`no op mapping for ${name}`);
}

/** Applies the vector's own comparison policy to the replayed output. */
function assertMatchesPolicy(name, vector, got) {
  const want = vector.output.map(bitsToFloat);
  assert.equal(got.length, want.length, name);
  if (vector.exact) {
    got.forEach((g, i) => assert.equal(floatToBits(g), vector.output[i], name));
  } else {
    const tolAbs = bitsToFloat(vector.tol_abs);
    const tolRel = bitsToFloat(vector.tol_rel);
    got.forEach((g, i) => {
      assert.ok(!Number.isNaN(g), name);
      assert.ok(Math.abs(g - want[i]) <= Math.max(tolAbs, tolRel * Math.abs(want[i])), name);
    });
  }
}

test("cdylib is discoverable", () => {
  assert.ok(fs.statSync(findCdylib()).isFile());
});

for (const [name, vector] of Object.entries(REFERENCE)) {
  test(`reference vector ${name} is reproduced`, () => {
    const x = vector.input.map(bitsToFloat);
    assertMatchesPolicy(name, vector, runOp(name, x));
  });
}

test("det3.standard is pinned in test code", () => {
  // The full det3.standard vector, pinned literally (rust-derived);
  // this test fails loudly even if reference.json were regenerated
  // wrongly.
  const x = [
    "4018000000000000", "3ff0000000000000", "3ff0000000000000",
    "4010000000000000", "c000000000000000", "4014000000000000",
    "4000000000000000", "4020000000000000", "401c000000000000",
  ].map(bitsToFloat);
  assert.equal(floatToBits(det3(x)), "c073200000000000");
});

test("fft.n8 first bins land within tolerance", () => {
  // One approx vector pinned in test code: the first four fft.n8 bins
  // must land within the recorded budgets.
  const x = [
    "3ff0000000000000", "3fe0000000000000", "bfe0000000000000", "4000000000000000",
    "3fd0000000000000", "bff8000000000000", "3ffc000000000000", "0000000000000000",
    "c002000000000000", "3fe0000000000000", "3fc0000000000000", "4008000000000000",
    "bfe8000000000000", "3ff0000000000000", "4004000000000000", "bff0000000000000",
  ].map(bitsToFloat);
  const want = ["4001000000000000", "4012000000000000", "3fead413cccfe77a", "bff712318007c2b1"].map(bitsToFloat);
  const tolAbs = bitsToFloat("3d3c25c268497682");
  const tolRel = bitsToFloat("3d719799812dea11");
  const got = fft(x);
  for (let i = 0; i < 4; i++) {
    assert.ok(!Number.isNaN(got[i]));
    assert.ok(Math.abs(got[i] - want[i]) <= Math.max(tolAbs, tolRel * Math.abs(want[i])));
  }
});

test("fft with an odd complex count is refused", () => {
  // 6 f64s = 3 complex pairs; 3 is not a power of two.
  assert.throws(() => fft([0, 0, 0, 0, 0, 0]), (err) => {
    assert.ok(err instanceof FfiError);
    assert.equal(err.status, -1);
    return true;
  });
});

test("dct2_2d geometry mismatch is refused", () => {
  assert.throws(() => dct2_2d(new Array(64).fill(0), 4, 8), (err) => {
    assert.ok(err instanceof FfiError);
    assert.equal(err.status, -1);
    return true;
  });
});

test("singular solve3 is rejected", () => {
  assert.throws(() => solve3(new Array(9).fill(0), new Array(3).fill(0)), (err) => {
    assert.ok(err instanceof FfiError);
    assert.equal(err.status, -2);
    return true;
  });
});

test("median of an empty input is refused", () => {
  assert.throws(() => median([]), (err) => {
    assert.ok(err instanceof FfiError);
    assert.equal(err.status, -1);
    return true;
  });
});

test("null data pointer is refused, not crashing", () => {
  // null is handed through as a NULL data pointer; the cdylib must
  // answer with a status code, never a crash.
  assert.throws(() => dct2(null), (err) => {
    assert.ok(err instanceof FfiError);
    assert.equal(err.status, -1);
    return true;
  });
});
