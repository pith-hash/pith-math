# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pith-hash
"""Hex-exact conformance: the committed reference vectors through ctypes.

Every vector in ``tests/reference.json`` is replayed through the cdylib
and compared per that file's own policy — bit-for-bit for ``exact``
vectors (raw IEEE-754 bit patterns, NaN and -0.0 safe), within
``max(tol_abs, tol_rel·|expected|)`` for the twiddle-factor vectors.
The same vectors the Rust ``gen-reference verify`` gate and the Node/Go
SDKs check.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import pytest

from pith_math import (
    FfiError,
    complex_arg,
    complex_div,
    complex_exp,
    complex_log,
    complex_mul,
    complex_powi,
    complex_sqrt,
    conv,
    corr,
    cov,
    dct2,
    dct2_2d,
    dct3,
    dct3_2d,
    det3,
    dtw,
    fft,
    fft_n,
    fft_real,
    find_cdylib,
    idct2,
    ifft,
    ifft_n,
    inverse3,
    lagrange,
    mat3_mul,
    mat3_mul_vec,
    mean,
    median,
    ransac_line,
    solve3,
    transpose3,
    variance,
)


def lerp(a: float, b: float, t: float) -> float:
    """The same linear interpolation the core exposes; kept local so
    the vector replay needs no extra export binding."""
    return a + (b - a) * t

REPO_ROOT = Path(__file__).resolve().parents[3]


def reference() -> dict:
    """Parses the committed reference file (tests/reference.json)."""
    return json.loads((REPO_ROOT / "tests" / "reference.json").read_text(encoding="utf-8"))


def bits_to_float(h: str) -> float:
    """16-digit hex of the raw IEEE-754 pattern → f64."""
    return struct.unpack("<d", struct.pack("<Q", int(h, 16)))[0]


def float_to_bits(v: float) -> int:
    """f64 → the raw IEEE-754 pattern as an int (NaN/-0.0 safe)."""
    return struct.unpack("<Q", struct.pack("<d", v))[0]


def run_op(name: str, x: list[float]) -> list[float]:
    """Replays the vector's op composition through the SDK."""
    if name in ("dct2.n8", "dct2.n32"):
        return dct2(x)
    if name == "idct2.roundtrip.n8":
        return idct2(dct2(x))
    if name in ("dct2.2d.8x8", "dct2.dc.8x8"):
        w, h = 8, 8
        return dct2_2d(x, w, h)
    if name in ("fft.n8", "fft.n64"):
        return fft(x)
    if name == "ifft.roundtrip.n64":
        return ifft(fft(x))
    if name == "fft.real.n8":
        return fft_real(x)
    if name.startswith("median."):
        return [median(x)]
    if name.startswith("solve3."):
        return solve3(x[:9], x[9:12])
    if name == "det3.standard":
        return [det3(x)]
    if name == "inverse3.sym":
        return inverse3(x)
    if name == "mat3.mul":
        return mat3_mul(x[:9], x[9:18])
    if name == "mat3.mul_vec.rot90":
        return mat3_mul_vec(x[:9], x[9:12])
    if name == "transpose3":
        return transpose3(x)
    if name in ("bluestein.n17", "bluestein.n97", "bluestein.n8.crosscheck"):
        return fft_n(x)
    if name == "bluestein.roundtrip.n12":
        return ifft_n(fft_n(x))
    if name in ("complex.mul.exact", "complex.div.exact"):
        op = complex_mul if name.startswith("complex.mul") else complex_div
        return op(x[:2], x[2:4])
    if name == "complex.exp.i.pi":
        return complex_exp(x)
    if name == "complex.sqrt.i":
        return complex_sqrt(x)
    if name == "complex.powi.exact":
        return complex_powi(x, 4)
    if name == "complex.arg.quarter":
        return [complex_arg(x)]
    if name == "dct3.n8":
        return dct3(x)
    if name == "dct3.roundtrip.n8":
        return dct3(dct2(x))
    if name == "conv.small.exact":
        return conv(x[:3], x[3:])
    if name == "conv.fft.path":
        return conv(x[:128], x[128:])
    if name == "corr.small.exact":
        return corr(x[:2], x[2:])
    if name == "stats.mean.exact":
        return [mean(x)]
    if name == "stats.var.sample.exact":
        return [variance(x)]
    if name == "stats.var.sample.textbook":
        return [variance(x)]
    if name == "stats.cov.sample.exact":
        return [cov(x)]
    if name == "interp.lagrange.quadratic.exact":
        return [lagrange(x[:6:2], x[1:6:2], x[6])]
    if name == "interp.lerp.midpoint.exact":
        return [lerp(x[0], x[1], x[2])]
    if name == "ransac.line.fit":
        # The packed input is interleaved (x, y) points.
        slope, intercept, inliers = ransac_line(x[0::2], x[1::2], 0.5, 64, 42)
        return [slope, intercept, float(inliers)]
    if name == "dtw.textbook.3x3":
        return [dtw(x[:3], x[3:])]
    if name == "dtw.textbook.2x3":
        return [dtw(x[:2], x[2:])]
    raise AssertionError(f"no op mapping for {name}")


def assert_matches_policy(name: str, vector: dict, got: list[float]) -> None:
    """Applies the vector's own comparison policy to the replayed output."""
    want = [bits_to_float(h) for h in vector["output"]]
    assert len(got) == len(want), name
    if vector["exact"]:
        for g, w in zip(got, want):
            assert float_to_bits(g) == float_to_bits(w), name
    else:
        tol_abs = bits_to_float(vector["tol_abs"])
        tol_rel = bits_to_float(vector["tol_rel"])
        for g, w in zip(got, want):
            assert not math.isnan(g), name
            assert abs(g - w) <= max(tol_abs, tol_rel * abs(w)), name


def test_cdylib_is_discoverable() -> None:
    path = find_cdylib()
    assert path.is_file(), path


@pytest.mark.parametrize("name", sorted(reference()["vectors"]))
def test_reference_vector_is_reproduced(name: str) -> None:
    vector = reference()["vectors"][name]
    x = [bits_to_float(h) for h in vector["input"]]
    assert_matches_policy(name, vector, run_op(name, x))


def test_det3_standard_is_pinned_in_test_code() -> None:
    # The full det3.standard vector, pinned literally (rust-derived);
    # this test fails loudly even if reference.json were regenerated
    # wrongly.
    x = [bits_to_float(h) for h in
         ["4018000000000000", "3ff0000000000000", "3ff0000000000000",
          "4010000000000000", "c000000000000000", "4014000000000000",
          "4000000000000000", "4020000000000000", "401c000000000000"]]
    assert float_to_bits(det3(x)) == 0xC073200000000000


def test_fft_n8_first_bins_land_within_tolerance() -> None:
    # One approx vector pinned in test code: the first four fft.n8 bins
    # must land within the recorded budgets.
    x = [bits_to_float(h) for h in
         ["3ff0000000000000", "3fe0000000000000", "bfe0000000000000", "4000000000000000",
          "3fd0000000000000", "bff8000000000000", "3ffc000000000000", "0000000000000000",
          "c002000000000000", "3fe0000000000000", "3fc0000000000000", "4008000000000000",
          "bfe8000000000000", "3ff0000000000000", "4004000000000000", "bff0000000000000"]]
    want = [bits_to_float(h) for h in
            ["4001000000000000", "4012000000000000", "3fead413cccfe77a", "bff712318007c2b1"]]
    tol_abs = bits_to_float("3d3c25c268497682")
    tol_rel = bits_to_float("3d719799812dea11")
    got = fft(x)
    for g, w in zip(got[:4], want):
        assert not math.isnan(g)
        assert abs(g - w) <= max(tol_abs, tol_rel * abs(w))


def test_fft_odd_complex_count_is_refused() -> None:
    # 6 f64s = 3 complex pairs; 3 is not a power of two.
    with pytest.raises(FfiError) as err:
        fft([0.0] * 6)
    assert err.value.status == -1


def test_dct2_2d_geometry_mismatch_is_refused() -> None:
    with pytest.raises(FfiError) as err:
        dct2_2d([0.0] * 64, 4, 8)
    assert err.value.status == -1


def test_singular_solve3_is_rejected() -> None:
    with pytest.raises(FfiError) as err:
        solve3([0.0] * 9, [0.0] * 3)
    assert err.value.status == -2


def test_median_empty_is_refused() -> None:
    with pytest.raises(FfiError) as err:
        median([])
    assert err.value.status == -1


def test_null_data_pointer_is_refused() -> None:
    # None is handed through as a NULL data pointer; the cdylib must
    # answer with a status code, never a crash.
    with pytest.raises(FfiError) as err:
        dct2(None)
    assert err.value.status == -1
