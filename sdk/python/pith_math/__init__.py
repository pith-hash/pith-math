# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pith-hash
"""pith-math SDK: FFT, DCT-II, median and 3×3 linear algebra through ctypes.

The single Rust core (the ``pith-math`` cdylib built by
``cargo build --release``) is loaded at runtime; this package carries
no third-party dependency — ``ctypes`` is the standard library.

Discovery order (the suite's cdylib convention):

1. ``PITH_CDYLIB`` — an explicit cdylib *file* path;
2. ``PITH_CDYLIB_DIR`` — a *directory* scanned for the cdylib names
   (the CD pipeline points this at ``target/release``);
3. the package directory itself (the built wheel ships the cdylib as
   package data);
4. ``<repo root>/target/release`` — the repository working-tree layout,
   so a source checkout runs against a local cargo build with no
   configuration.

Every transform takes and returns plain ``list[float]`` buffers; the
scalar order statistics (:func:`median`, :func:`det3`) return a
``float``. Non-zero FFI statuses raise :class:`FfiError` carrying the
raw status code — the cdylib never panics through this boundary.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

__all__ = [
    "FfiError",
    "LibraryNotFoundError",
    "find_cdylib",
    "dct2",
    "idct2",
    "dct2_2d",
    "fft",
    "ifft",
    "fft_real",
    "solve3",
    "det3",
    "inverse3",
    "transpose3",
    "mat3_mul",
    "mat3_mul_vec",
    "median",
    "complex_div",
    "complex_exp",
    "complex_log",
    "complex_mul",
    "complex_powi",
    "complex_sqrt",
    "complex_arg",
    "fft_n",
    "ifft_n",
    "dct3",
    "dct3_2d",
    "conv",
    "corr",
    "mean",
    "variance",
    "cov",
    "lagrange",
    "ransac_line",
    "dtw",
    "STATUS_OK",
    "STATUS_INVALID",
    "STATUS_REJECTED",
]

#: Status: success.
STATUS_OK = 0
#: Status: a caller argument is invalid (null pointer, empty input,
#: odd/non-power-of-two transform length, geometry mismatch).
STATUS_INVALID = -1
#: Status: the core refused the input (a singular matrix).
STATUS_REJECTED = -2

#: Every cdylib file name cargo may drop into the build directory, per
#: platform (windows / linux / macOS).
CDYLIB_NAMES = ("pith_math.dll", "libpith_math.so", "libpith_math.dylib")


class LibraryNotFoundError(OSError):
    """No cdylib was found through the discovery chain."""


class FfiError(Exception):
    """A non-zero status code came back from the cdylib."""

    def __init__(self, op: str, status: int) -> None:
        kind = {
            STATUS_INVALID: "invalid argument",
            STATUS_REJECTED: "input rejected",
        }.get(status, "unknown failure")
        super().__init__(f"{op} failed: {kind} (status {status})")
        #: The raw status code the FFI returned.
        self.status = status


def find_cdylib() -> Path:
    """Locates the cdylib through the suite's discovery chain."""
    explicit = os.environ.get("PITH_CDYLIB")
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
    env_dir = os.environ.get("PITH_CDYLIB_DIR")
    candidates: list[Path] = []
    if env_dir:
        env_dir_path = Path(env_dir)
        candidates.append(env_dir_path)
        if not env_dir_path.is_absolute():
            # CD and local runs invoke tools from the repository root or
            # from sdk/<lang>; resolve the env value against both.
            candidates.append(Path.cwd() / env_dir_path)
            candidates.append(Path(__file__).resolve().parents[3] / env_dir_path)
    candidates.append(Path(__file__).resolve().parent)  # packaged wheel
    candidates.append(Path(__file__).resolve().parents[3] / "target" / "release")
    for directory in candidates:
        for name in CDYLIB_NAMES:
            p = directory / name
            if p.is_file():
                return p
    raise LibraryNotFoundError(
        "no pith-math cdylib found (searched PITH_CDYLIB, PITH_CDYLIB_DIR, "
        "the package directory and <repo>/target/release); "
        "run `cargo build --release` first"
    )


_lib: ctypes.CDLL | None = None


def _bind_alloc(lib: ctypes.CDLL, name: str, two_d: bool = False) -> None:
    fn = getattr(lib, name)
    fn.argtypes = [
        ctypes.c_void_p,  # input
        ctypes.c_size_t,  # in_len
        *([ctypes.c_size_t, ctypes.c_size_t] if two_d else []),  # w, h
        ctypes.POINTER(ctypes.c_void_p),  # out buffer
        ctypes.POINTER(ctypes.c_size_t),  # out length (bytes)
    ]
    fn.restype = ctypes.c_int32


def _bind_scalar(lib: ctypes.CDLL, name: str) -> None:
    fn = getattr(lib, name)
    fn.argtypes = [
        ctypes.c_void_p,  # input
        ctypes.c_size_t,  # in_len
        ctypes.POINTER(ctypes.c_double),  # out slot
    ]
    fn.restype = ctypes.c_int32


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(str(find_cdylib()))
        for name in (
            "pith_math_dct2",
            "pith_math_idct2",
            "pith_math_fft",
            "pith_math_ifft",
            "pith_math_fft_real",
            "pith_math_solve3",
            "pith_math_inverse3",
            "pith_math_transpose3",
            "pith_math_mat3_mul",
            "pith_math_mat3_mul_vec",
            "pith_math_complex_div",
            "pith_math_complex_exp",
            "pith_math_complex_log",
            "pith_math_complex_mul",
            "pith_math_complex_powi",
            "pith_math_complex_sqrt",
            "pith_math_fft_n",
            "pith_math_ifft_n",
            "pith_math_dct3",
            "pith_math_dct3_2d",
        ):
            _bind_alloc(lib, name)
        _bind_alloc(lib, "pith_math_dct2_2d", two_d=True)
        _bind_alloc(lib, "pith_math_dct3_2d", two_d=True)
        for name in ("pith_math_det3", "pith_math_median", "pith_math_mean",
                     "pith_math_var", "pith_math_cov", "pith_math_complex_arg"):
            _bind_scalar(lib, name)
        # Packed two-operand kernels: (ptr, len, a_len, out**, out_len*).
        for name in ("pith_math_conv", "pith_math_corr", "pith_math_dtw"):
            fn = getattr(lib, name)
            fn.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t),
            ] if name != "pith_math_dtw" else [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
            ]
            fn.restype = ctypes.c_int32
        # Lagrange: (ptr, len, x: f64, out*).
        fn = lib.pith_math_lagrange
        fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_double,
                       ctypes.POINTER(ctypes.c_double)]
        fn.restype = ctypes.c_int32
        # complex_powi: (ptr, len, n, out**, out_len*).
        fn = lib.pith_math_complex_powi
        fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                       ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
        fn.restype = ctypes.c_int32
        # RANSAC: (ptr, len, threshold, iterations, seed, out**, out_len*).
        fn = lib.pith_math_ransac_line
        fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_double,
                       ctypes.c_size_t, ctypes.c_uint64,
                       ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
        fn.restype = ctypes.c_int32
        lib.pith_math_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.pith_math_free.restype = None
        _lib = lib
    return _lib


def _pack(vals) -> ctypes.Array:
    """Packs a sequence of floats into a contiguous ``f64`` buffer;
    ``None`` passes through as a NULL pointer."""
    if vals is None:
        return None
    arr = [float(v) for v in vals]
    return (ctypes.c_double * len(arr))(*arr)


def _call_alloc(op_name: str, arr, *extra: int) -> list[float]:
    """Runs one allocating op: hands ownership back, copies the buffer
    out (element count = bytes / 8) and releases it."""
    fn = getattr(_load(), op_name)
    out = ctypes.c_void_p()
    out_len = ctypes.c_size_t()
    buf = _pack(arr)
    n = 0 if buf is None else len(buf)
    status = fn(buf, n, *extra, ctypes.byref(out), ctypes.byref(out_len))
    if status != STATUS_OK:
        raise FfiError(op_name, status)
    try:
        count = out_len.value // ctypes.sizeof(ctypes.c_double)
        return list(ctypes.cast(out, ctypes.POINTER(ctypes.c_double))[:count])
    finally:
        _load().pith_math_free(out, out_len.value)


def _call_scalar(op_name: str, arr) -> float:
    fn = getattr(_load(), op_name)
    buf = _pack(arr)
    n = 0 if buf is None else len(buf)
    out = ctypes.c_double()
    status = fn(buf, n, ctypes.byref(out))
    if status != STATUS_OK:
        raise FfiError(op_name, status)
    return out.value


def dct2(x) -> list[float]:
    """1D orthonormal DCT-II. An empty input raises
    :class:`FfiError` with ``status == STATUS_INVALID``."""
    return _call_alloc("pith_math_dct2", x)


def idct2(x) -> list[float]:
    """1D orthonormal DCT-III — the exact inverse of :func:`dct2`."""
    return _call_alloc("pith_math_idct2", x)


def dct2_2d(x, w: int, h: int) -> list[float]:
    """Separable 2D orthonormal DCT-II over a ``w × h`` row-major
    matrix; ``len(x)`` must equal ``w * h``."""
    return _call_alloc("pith_math_dct2_2d", x, w, h)


def fft(x) -> list[float]:
    """Forward DFT over interleaved complex pairs ``[re, im, …]``; the
    complex count must be a non-zero power of two."""
    return _call_alloc("pith_math_fft", x)


def ifft(x) -> list[float]:
    """Inverse DFT, the exact (within ulps) inverse of :func:`fft`."""
    return _call_alloc("pith_math_ifft", x)


def fft_real(x) -> list[float]:
    """Forward DFT of a real signal; returns the full ``2·len(x)``
    interleaved spectrum. ``len(x)`` must be a non-zero power of two."""
    return _call_alloc("pith_math_fft_real", x)


def solve3(a, b) -> list[float]:
    """Solves the 3×3 system ``a·x = b`` (row-major ``a``). A singular
    system raises :class:`FfiError` with ``status == STATUS_REJECTED``
    — ordinary RANSAC data, not a caller bug."""
    return _call_alloc("pith_math_solve3", list(a) + list(b))


def det3(m) -> float:
    """Determinant of the row-major 3×3 matrix ``m``."""
    return _call_scalar("pith_math_det3", m)


def inverse3(m) -> list[float]:
    """Inverse of the row-major 3×3 matrix ``m``; singular input
    raises :class:`FfiError` with ``status == STATUS_REJECTED``."""
    return _call_alloc("pith_math_inverse3", m)


def transpose3(m) -> list[float]:
    """Transpose of the row-major 3×3 matrix ``m``."""
    return _call_alloc("pith_math_transpose3", m)


def mat3_mul(a, b) -> list[float]:
    """Matrix product ``a·b`` of two row-major 3×3 factors."""
    return _call_alloc("pith_math_mat3_mul", list(a) + list(b))


def mat3_mul_vec(m, v) -> list[float]:
    """Matrix-vector product ``m·v``."""
    return _call_alloc("pith_math_mat3_mul_vec", list(m) + list(v))


def median(x) -> float:
    """The median of ``x`` — the lower middle element for even ``n``,
    never the mean of the two middles. Empty input raises
    :class:`FfiError` with ``status == STATUS_INVALID``."""
    return _call_scalar("pith_math_median", x)


# ---------------------------------------------------------------------
# Tier-1 expansion
# ---------------------------------------------------------------------


def complex_mul(z, w) -> list[float]:
    """Complex product of two interleaved pairs ``[re, im]``."""
    return _call_alloc("pith_math_complex_mul", list(z) + list(w))


def complex_div(z, w) -> list[float]:
    """Complex quotient ``z / w`` (Smith's scaled algorithm); a zero
    denominator raises :class:`FfiError` with ``STATUS_REJECTED``."""
    return _call_alloc("pith_math_complex_div", list(z) + list(w))


def complex_exp(z) -> list[float]:
    """``e^z`` for the interleaved pair ``z``."""
    return _call_alloc("pith_math_complex_exp", z)


def complex_log(z) -> list[float]:
    """Principal-branch natural logarithm ``[ln|z|, arg z]``; ``z = 0``
    raises :class:`FfiError` with ``STATUS_REJECTED``."""
    return _call_alloc("pith_math_complex_log", z)


def complex_sqrt(z) -> list[float]:
    """Principal square root of the interleaved pair ``z``."""
    return _call_alloc("pith_math_complex_sqrt", z)


def complex_powi(z, n: int) -> list[float]:
    """Integer power by exponentiation-by-squaring; ``n`` may be zero
    (exactly one) or negative (reciprocal through ``n``'s absolute
    value, per the C surface's ``usize`` exponent)."""
    fn = getattr(_load(), "pith_math_complex_powi")
    out = ctypes.c_void_p()
    out_len = ctypes.c_size_t()
    buf = _pack(z)
    status = fn(buf, 0 if buf is None else len(buf), n if n >= 0 else 0,
                ctypes.byref(out), ctypes.byref(out_len))
    if status != STATUS_OK:
        raise FfiError("pith_math_complex_powi", status)
    try:
        count = out_len.value // ctypes.sizeof(ctypes.c_double)
        return list(ctypes.cast(out, ctypes.POINTER(ctypes.c_double))[:count])
    finally:
        _load().pith_math_free(out, out_len.value)


def complex_arg(z) -> float:
    """The principal argument of ``z`` in radians, ``(-π, π]``."""
    return _call_scalar("pith_math_complex_arg", z)


def fft_n(x) -> list[float]:
    """Forward DFT of **any** ``n ≥ 1`` (Bluestein below the radix-2
    sizes): interleaved complex pairs in, interleaved spectrum out."""
    return _call_alloc("pith_math_fft_n", x)


def ifft_n(x) -> list[float]:
    """Inverse of :func:`fft_n`."""
    return _call_alloc("pith_math_ifft_n", x)


def dct3(x) -> list[float]:
    """Orthonormal DCT-III — the exact inverse (transposed kernel) of
    :func:`dct2`."""
    return _call_alloc("pith_math_dct3", x)


def dct3_2d(x, w: int, h: int) -> list[float]:
    """Separable 2D orthonormal DCT-III over a ``w × h`` row-major
    matrix; the exact inverse of :func:`dct2_2d`."""
    return _call_alloc("pith_math_dct3_2d", x, w, h)


def conv(x, k) -> list[float]:
    """Linear convolution of ``x`` with kernel ``k`` — full mode,
    ``len`` ``= len(x) + len(k) − 1`` (direct below 256 elements,
    FFT above)."""
    return _call_alloc("pith_math_conv", list(x) + list(k), len(x))


def corr(x, k) -> list[float]:
    """Cross-correlation — :func:`conv` with the flipped kernel."""
    return _call_alloc("pith_math_corr", list(x) + list(k), len(x))


def mean(x) -> float:
    """Arithmetic mean; empty input raises :class:`FfiError` with
    ``STATUS_INVALID``."""
    return _call_scalar("pith_math_mean", x)


def variance(x) -> float:
    """Population variance (``1/n`` normalisation)."""
    return _call_scalar("pith_math_var", x)


def cov(x) -> float:
    """Covariance of the paired samples in ``x`` (interleaved pairs);
    fewer than two pairs raises ``STATUS_INVALID``."""
    return _call_scalar("pith_math_cov", x)


def lagrange(xs, ys, x: float) -> float:
    """Lagrange interpolation through ``(xs[i], ys[i])`` evaluated at
    ``x``. Degenerate node sets (duplicate x-coordinates, mismatched
    lengths) raise :class:`FfiError` with ``STATUS_INVALID``."""
    if len(xs) != len(ys):
        raise FfiError("pith_math_lagrange", STATUS_INVALID)
    packed = [v for pair in zip(xs, ys) for v in pair]
    fn = getattr(_load(), "pith_math_lagrange")
    buf = _pack(packed)
    out = ctypes.c_double()
    status = fn(buf, 0 if buf is None else len(buf), float(x), ctypes.byref(out))
    if status != STATUS_OK:
        raise FfiError("pith_math_lagrange", status)
    return out.value


def ransac_line(xs, ys, threshold: float, iterations: int, seed: int):
    """Seeded RANSAC line fit: returns ``(slope, intercept, inliers)``
    where ``inliers`` counts points within ``threshold`` of the model,
    or ``None`` when no two-point sample produced a usable line."""
    if len(xs) != len(ys):
        raise FfiError("pith_math_ransac_line", STATUS_INVALID)
    packed = [v for pair in zip(xs, ys) for v in pair]
    fn = getattr(_load(), "pith_math_ransac_line")
    buf = _pack(packed)
    out = ctypes.c_void_p()
    out_len = ctypes.c_size_t()
    status = fn(buf, 0 if buf is None else len(buf), float(threshold),
                int(iterations), int(seed), ctypes.byref(out), ctypes.byref(out_len))
    if status != STATUS_OK:
        raise FfiError("pith_math_ransac_line", status)
    try:
        count = out_len.value // ctypes.sizeof(ctypes.c_double)
        vals = list(ctypes.cast(out, ctypes.POINTER(ctypes.c_double))[:count])
    finally:
        _load().pith_math_free(out, out_len.value)
    return vals[0], vals[1], int(vals[2])


def dtw(a, b) -> float:
    """Dynamic time warping distance (absolute-difference local cost,
    three monotone steps, optimal path, no window constraint); empty
    input raises :class:`FfiError` with ``STATUS_INVALID``."""
    fn = getattr(_load(), "pith_math_dtw")
    buf = _pack(list(a) + list(b))
    out = ctypes.c_double()
    status = fn(buf, 0 if buf is None else len(buf), len(a), ctypes.byref(out))
    if status != STATUS_OK:
        raise FfiError("pith_math_dtw", status)
    return out.value
