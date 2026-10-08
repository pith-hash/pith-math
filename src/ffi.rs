//! The C ABI surface of `pith-math`: the entry points the Python
//! (ctypes), Node (koffi) and Go (cgo) SDKs bind through.
//!
//! The suite's FFI convention, defined by this module and mirrored by
//! every `pith-*` cdylib:
//!
//! * one flat set of `#[unsafe(no_mangle)] pub unsafe extern "C"`
//!   functions — raw pointers plus lengths, no structs across the
//!   boundary;
//! * every function returns a status code (see the constants below),
//!   never a `Result`, never a panic: the panicking preconditions of
//!   the core (`fft`'s power-of-two length, `dct2_2d`'s geometry,
//!   empty inputs) are pre-validated here and reported as
//!   [`PITH_E_INVALID`] instead;
//! * an operation either hands ownership of a freshly allocated
//!   buffer to the caller (with [`pith_math_free`]) or writes a
//!   scalar through a caller-provided out-slot ([`pith_math_det3`],
//!   [`pith_math_median`]) — nothing else crosses the boundary;
//! * the `unsafe` allowance is confined to this module; every core
//!   module stays unsafe-free behind the crate-root `#![deny]`.
//!
//! Buffer lengths are **byte** counts: `out_len` receives the size in
//! bytes and `pith_math_free` takes the same byte count back, so a
//! foreign caller never multiplies by `sizeof(f64)` except to derive
//! an element count (`bytes / 8`).

#![allow(unsafe_code)]

use crate::bluestein::{fft_arbitrary, ifft_arbitrary};
use crate::complex::Complex;
use crate::conv::{convolve, correlate};
use crate::dct::{dct2, dct2_2d, idct2};
use crate::dct3::{dct3, dct3_2d};
use crate::dtw::dtw_distance;
use crate::fft::{fft, fft_real, ifft};
use crate::interp::lagrange_eval;
use crate::linalg::{det3, inverse3, mat3_mul, mat3_mul_vec, transpose3};
use crate::median::median_copy;
use crate::ransac::ransac_line;
use crate::solve3::solve3;
use crate::stats::{covariance, mean, variance};

/// Status: success.
pub const PITH_OK: i32 = 0;
/// Status: a caller argument is invalid — a null pointer, an empty
/// input, an odd or non-power-of-two complex count, a non-power-of-two
/// real length, or a length that disagrees with the declared geometry.
pub const PITH_E_INVALID: i32 = -1;
/// Status: the core refused the input — a singular matrix for
/// [`pith_math_solve3`] or [`pith_math_inverse3`].
pub const PITH_E_REJECTED: i32 = -2;

/// Hands a freshly allocated result to the caller: the buffer address
/// through `out`, its size in bytes through `out_len`, ownership
/// included. The boxed slice remembers its exact length, which
/// [`pith_math_free`] reconstructs from the same byte count.
///
/// # Safety
///
/// `out` and `out_len` must point to writable memory; both are written
/// exactly once.
unsafe fn hand_out(out: *mut *mut f64, out_len: *mut usize, result: Vec<f64>) -> i32 {
    let bytes = result.len() * core::mem::size_of::<f64>();
    let ptr = Box::into_raw(result.into_boxed_slice());
    unsafe {
        *out = ptr.cast::<f64>();
        *out_len = bytes;
    }
    PITH_OK
}

/// Null-guard shared by every export: [`PITH_E_INVALID`] when a data
/// pointer or an out-slot is missing.
fn null_status(input: *const f64, out: *mut *mut f64, out_len: *mut usize) -> Option<i32> {
    if input.is_null() || out.is_null() || out_len.is_null() {
        return Some(PITH_E_INVALID);
    }
    None
}

/// The scalar-out variant of [`null_status`].
fn null_status_scalar(input: *const f64, out: *mut f64) -> Option<i32> {
    if input.is_null() || out.is_null() {
        return Some(PITH_E_INVALID);
    }
    None
}

/// Runs one scalar-out kernel through the validate → compute →
/// write-slot pipeline shared by every non-allocating export.
///
/// # Safety
///
/// `input` must point to `in_len` readable `f64`s and `out` to one
/// writable `f64`; both must stay valid for the duration of the call.
unsafe fn run_scalar(
    input: *const f64,
    in_len: usize,
    out: *mut f64,
    kernel: fn(&[f64]) -> Result<f64, i32>,
) -> i32 {
    if let Some(status) = null_status_scalar(input, out) {
        return status;
    }
    let x = unsafe { core::slice::from_raw_parts(input, in_len) };
    match kernel(x) {
        Ok(value) => {
            unsafe { *out = value };
            PITH_OK
        }
        Err(status) => status,
    }
}

/// Runs one of the flat-array kernels through the standard
/// validate → compute → hand-out pipeline shared by every allocating
/// export.
///
/// # Safety
///
/// `input` must point to `in_len` readable `f64`s, `out` to one
/// writable pointer and `out_len` to one writable `usize`; all must
/// stay valid for the duration of the call.
unsafe fn run_alloc(
    input: *const f64,
    in_len: usize,
    out: *mut *mut f64,
    out_len: *mut usize,
    kernel: fn(&[f64]) -> Result<Vec<f64>, i32>,
) -> i32 {
    if input.is_null() || out.is_null() || out_len.is_null() {
        return PITH_E_INVALID;
    }
    let x = unsafe { core::slice::from_raw_parts(input, in_len) };
    match kernel(x) {
        Ok(result) => unsafe { hand_out(out, out_len, result) },
        Err(status) => status,
    }
}

/// Reassembles a row-major flat buffer into the core's `[[f64; 3]; 3]`.
/// Every caller validates the length (9, or the 9-prefix of a longer
/// buffer) before calling, so indexing stays in bounds.
fn mat3(flat: &[f64]) -> [[f64; 3]; 3] {
    core::array::from_fn(|r| core::array::from_fn(|c| flat[r * 3 + c]))
}

/// Flattens a matrix back to row-major `f64`s.
fn mat3_flat(m: &[[f64; 3]; 3]) -> Vec<f64> {
    m.iter().flat_map(|row| row.iter().copied()).collect()
}

/// Flattens complex bins to the interleaved `[re, im, …]` wire form.
fn flat_complex(cs: &[Complex]) -> Vec<f64> {
    cs.iter().flat_map(|c| [c.re, c.im]).collect()
}

/// Validates the interleaved-complex layout (`in_len` even, half the
/// elements a non-zero power of two — exactly what [`fft`] and
/// [`ifft`] assert) and runs `kernel`.
fn complex_kernel(flat: &[f64], kernel: fn(&mut [Complex])) -> Result<Vec<f64>, i32> {
    let half = flat.len() / 2;
    if flat.len() % 2 != 0 || !half.is_power_of_two() {
        return Err(PITH_E_INVALID);
    }
    let mut buf: Vec<Complex> = flat
        .chunks_exact(2)
        .map(|c| Complex::new(c[0], c[1]))
        .collect();
    kernel(&mut buf);
    Ok(flat_complex(&buf))
}

/// The safe core of [`pith_math_dct2`]. An empty input is a caller
/// bug: the transform of nothing is not a vector the kit uses.
fn dct2_core(x: &[f64]) -> Result<Vec<f64>, i32> {
    if x.is_empty() {
        return Err(PITH_E_INVALID);
    }
    Ok(dct2(x))
}

/// The safe core of [`pith_math_idct2`].
fn idct2_core(x: &[f64]) -> Result<Vec<f64>, i32> {
    if x.is_empty() {
        return Err(PITH_E_INVALID);
    }
    Ok(idct2(x))
}

/// The safe core of [`pith_math_dct2_2d`]: geometry first (the core
/// asserts `len == w·h` and positive dimensions), then the separable
/// transform on a private copy of the caller's read-only buffer.
fn dct2_2d_core(data: &[f64], w: usize, h: usize) -> Result<Vec<f64>, i32> {
    if w == 0 || h == 0 || Some(data.len()) != w.checked_mul(h) {
        return Err(PITH_E_INVALID);
    }
    let mut buf = data.to_vec();
    dct2_2d(&mut buf, w, h);
    Ok(buf)
}

/// The safe core of [`pith_math_fft`].
fn fft_core(flat: &[f64]) -> Result<Vec<f64>, i32> {
    complex_kernel(flat, fft)
}

/// The safe core of [`pith_math_ifft`].
fn ifft_core(flat: &[f64]) -> Result<Vec<f64>, i32> {
    complex_kernel(flat, ifft)
}

/// The safe core of [`pith_math_fft_real`]: a non-empty power-of-two
/// real length, expanded to the full `2n` interleaved spectrum.
fn fft_real_core(x: &[f64]) -> Result<Vec<f64>, i32> {
    if x.is_empty() || !x.len().is_power_of_two() {
        return Err(PITH_E_INVALID);
    }
    Ok(flat_complex(&fft_real(x)))
}

/// The safe core of [`pith_math_solve3`]: 9 row-major matrix entries
/// followed by the 3 right-hand side entries. A singular system is
/// [`PITH_E_REJECTED`] — data, not a caller bug.
fn solve3_core(flat: &[f64]) -> Result<Vec<f64>, i32> {
    if flat.len() != 12 {
        return Err(PITH_E_INVALID);
    }
    let a = mat3(&flat[..9]);
    let b = [flat[9], flat[10], flat[11]];
    match solve3(&a, &b) {
        Some(x) => Ok(x.to_vec()),
        None => Err(PITH_E_REJECTED),
    }
}

/// The safe core of [`pith_math_det3`].
fn det3_core(flat: &[f64]) -> Result<f64, i32> {
    if flat.len() != 9 {
        return Err(PITH_E_INVALID);
    }
    Ok(det3(&mat3(flat)))
}

/// The safe core of [`pith_math_inverse3`]; singular is rejected.
fn inverse3_core(flat: &[f64]) -> Result<Vec<f64>, i32> {
    if flat.len() != 9 {
        return Err(PITH_E_INVALID);
    }
    inverse3(&mat3(flat))
        .map(|m| mat3_flat(&m))
        .ok_or(PITH_E_REJECTED)
}

/// The safe core of [`pith_math_transpose3`].
fn transpose3_core(flat: &[f64]) -> Result<Vec<f64>, i32> {
    if flat.len() != 9 {
        return Err(PITH_E_INVALID);
    }
    Ok(mat3_flat(&transpose3(&mat3(flat))))
}

/// The safe core of [`pith_math_mat3_mul`]: two row-major factors
/// packed back to back.
fn mat3_mul_core(flat: &[f64]) -> Result<Vec<f64>, i32> {
    if flat.len() != 18 {
        return Err(PITH_E_INVALID);
    }
    Ok(mat3_flat(&mat3_mul(&mat3(&flat[..9]), &mat3(&flat[9..]))))
}

/// The safe core of [`pith_math_mat3_mul_vec`]: 9 matrix entries then
/// the 3-element vector.
fn mat3_mul_vec_core(flat: &[f64]) -> Result<Vec<f64>, i32> {
    if flat.len() != 12 {
        return Err(PITH_E_INVALID);
    }
    let v = [flat[9], flat[10], flat[11]];
    Ok(mat3_mul_vec(&mat3(&flat[..9]), &v).to_vec())
}

/// The safe core of [`pith_math_median`]: the same lower-middle order
/// statistic as the core's `median`, without mutating the caller's
/// buffer. An empty input is [`PITH_E_INVALID`].
fn median_core(x: &[f64]) -> Result<f64, i32> {
    median_copy(x).ok_or(PITH_E_INVALID)
}

// -- Tier-1 cores ------------------------------------------------------

/// The safe core of [`pith_math_fft_n`]: the interleaved-complex
/// layout with `n = in_len / 2 ≥ 1` — any length, not just powers of
/// two.
fn fft_n_core(flat: &[f64]) -> Result<Vec<f64>, i32> {
    if flat.len() % 2 != 0 || flat.is_empty() {
        return Err(PITH_E_INVALID);
    }
    let mut buf: Vec<Complex> = flat
        .chunks_exact(2)
        .map(|c| Complex::new(c[0], c[1]))
        .collect();
    fft_arbitrary(&mut buf);
    Ok(flat_complex(&buf))
}

/// The safe core of [`pith_math_dct3`].
fn dct3_core(x: &[f64]) -> Result<Vec<f64>, i32> {
    if x.is_empty() {
        return Err(PITH_E_INVALID);
    }
    Ok(dct3(x))
}

/// The safe core of [`pith_math_dct3_2d`]: geometry first, as in
/// [`dct2_2d_core`].
fn dct3_2d_core(data: &[f64], w: usize, h: usize) -> Result<Vec<f64>, i32> {
    if w == 0 || h == 0 || Some(data.len()) != w.checked_mul(h) {
        return Err(PITH_E_INVALID);
    }
    let mut buf = data.to_vec();
    dct3_2d(&mut buf, w, h);
    Ok(buf)
}

/// The safe core of [`pith_math_ifft_n`].
fn ifft_n_core(flat: &[f64]) -> Result<Vec<f64>, i32> {
    if flat.len() % 2 != 0 || flat.is_empty() {
        return Err(PITH_E_INVALID);
    }
    let mut buf: Vec<Complex> = flat
        .chunks_exact(2)
        .map(|c| Complex::new(c[0], c[1]))
        .collect();
    ifft_arbitrary(&mut buf);
    Ok(flat_complex(&buf))
}

/// The safe core of [`pith_math_conv`] / [`pith_math_corr`]: one
/// packed buffer, `a_len` the element count of the first operand.
fn packed_pair_core(
    flat: &[f64],
    a_len: usize,
    op: fn(&[f64], &[f64]) -> Vec<f64>,
) -> Result<Vec<f64>, i32> {
    if a_len == 0 || flat.len() <= a_len {
        return Err(PITH_E_INVALID);
    }
    let (a, b) = flat.split_at(a_len);
    Ok(op(a, b))
}

/// The safe core of [`pith_math_mean`].
fn mean_core(x: &[f64]) -> Result<f64, i32> {
    mean(x).ok_or(PITH_E_INVALID)
}

/// The safe core of [`pith_math_var`]: fewer than two observations is
/// a caller bug.
fn var_core(x: &[f64]) -> Result<f64, i32> {
    if x.len() < 2 {
        return Err(PITH_E_INVALID);
    }
    variance(x).ok_or(PITH_E_INVALID)
}

/// The safe core of [`pith_math_cov`]: one packed buffer split into
/// two equal-length halves, each of at least two observations.
fn cov_core(flat: &[f64]) -> Result<f64, i32> {
    if flat.len() % 2 != 0 || flat.len() < 4 {
        return Err(PITH_E_INVALID);
    }
    let (a, b) = flat.split_at(flat.len() / 2);
    covariance(a, b).ok_or(PITH_E_INVALID)
}

/// The safe core of [`pith_math_lagrange`]: `points` packed as
/// interleaved `(x, y)` pairs plus the evaluation abscissa. Duplicated
/// abscissae are [`PITH_E_REJECTED`] — degenerate data, not a bug.
fn lagrange_core(packed: &[f64], x: f64) -> Result<f64, i32> {
    if packed.len() % 2 != 0 || packed.is_empty() {
        return Err(PITH_E_INVALID);
    }
    let points: Vec<(f64, f64)> = packed.chunks_exact(2).map(|p| (p[0], p[1])).collect();
    lagrange_eval(&points, x).ok_or(PITH_E_REJECTED)
}

/// The safe core of [`pith_math_dtw`]: packed sequences split at
/// `a_len`, both non-empty.
fn dtw_core(packed: &[f64], a_len: usize) -> Result<f64, i32> {
    if a_len == 0 || packed.len() <= a_len {
        return Err(PITH_E_INVALID);
    }
    let (a, b) = packed.split_at(a_len);
    dtw_distance(a, b).ok_or(PITH_E_INVALID)
}

/// The safe core of [`pith_math_ransac_line`]: points packed as
/// interleaved `(x, y)` pairs. Configuration errors are
/// [`PITH_E_INVALID`]; a run that found no model at all (every sample
/// degenerate) is [`PITH_E_REJECTED`].
#[allow(clippy::too_many_arguments)]
fn ransac_core(
    packed: &[f64],
    threshold: f64,
    iterations: usize,
    seed: u64,
) -> Result<Vec<f64>, i32> {
    if packed.len() < 4 || packed.len() % 2 != 0 {
        return Err(PITH_E_INVALID);
    }
    let points: Vec<(f64, f64)> = packed.chunks_exact(2).map(|p| (p[0], p[1])).collect();
    if points.len() < 2 || iterations == 0 || threshold.is_nan() || threshold <= 0.0 {
        return Err(PITH_E_INVALID);
    }
    ransac_line(&points, threshold, iterations, seed)
        .map(|fit| vec![fit.slope, fit.intercept, fit.inliers as f64])
        .ok_or(PITH_E_REJECTED)
}

/// The safe core of the unary complex ops ([`pith_math_complex_exp`],
/// [`pith_math_complex_sqrt`]): exactly one interleaved pair in, one
/// pair out.
fn complex_unary_core(flat: &[f64], op: fn(Complex) -> Complex) -> Result<Vec<f64>, i32> {
    if flat.len() != 2 {
        return Err(PITH_E_INVALID);
    }
    let z = op(Complex::new(flat[0], flat[1]));
    Ok(vec![z.re, z.im])
}

/// The safe core of [`pith_math_complex_mul`]: exactly two interleaved
/// pairs in.
fn complex_mul_core(flat: &[f64]) -> Result<Vec<f64>, i32> {
    complex_binary_core(flat, |a, b| a * b)
}

/// The safe core of [`pith_math_complex_div`]: a zero denominator is
/// [`PITH_E_REJECTED`] — the domain refusal, like a singular matrix.
fn complex_div_core(flat: &[f64]) -> Result<Vec<f64>, i32> {
    if flat.len() == 4 && flat[2] == 0.0 && flat[3] == 0.0 {
        return Err(PITH_E_REJECTED);
    }
    complex_binary_core(flat, |a, b| a / b)
}

/// The safe core of the binary complex ops ([`pith_math_complex_mul`],
/// [`pith_math_complex_div`]): exactly two interleaved pairs in.
fn complex_binary_core(flat: &[f64], op: fn(Complex, Complex) -> Complex) -> Result<Vec<f64>, i32> {
    if flat.len() != 4 {
        return Err(PITH_E_INVALID);
    }
    let z = op(
        Complex::new(flat[0], flat[1]),
        Complex::new(flat[2], flat[3]),
    );
    Ok(vec![z.re, z.im])
}

/// The safe core of [`pith_math_complex_powi`].
fn complex_powi_core(flat: &[f64], n: usize) -> Result<Vec<f64>, i32> {
    if flat.len() != 2 || n > i32::MAX as usize {
        return Err(PITH_E_INVALID);
    }
    let z = Complex::new(flat[0], flat[1]).powi(n as i32);
    Ok(vec![z.re, z.im])
}

/// The safe core of [`pith_math_complex_arg`].
fn complex_arg_core(flat: &[f64]) -> Result<f64, i32> {
    if flat.len() != 2 {
        return Err(PITH_E_INVALID);
    }
    Ok(Complex::new(flat[0], flat[1]).arg())
}

/// 1D orthonormal DCT-II of `in_len` `f64`s.
///
/// `input` points at the samples; on success a freshly allocated
/// buffer of the same length is written through `out` with its size in
/// bytes through `out_len`, and the caller owns it (release with
/// [`pith_math_free`]). An empty input is [`PITH_E_INVALID`].
///
/// # Safety
///
/// `input` must point to `in_len` readable `f64`s; `out` and `out_len`
/// to writable slots. No pointer is retained.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_dct2(
    input: *const f64,
    in_len: usize,
    out: *mut *mut f64,
    out_len: *mut usize,
) -> i32 {
    unsafe { run_alloc(input, in_len, out, out_len, dct2_core) }
}

/// 1D orthonormal DCT-III — the exact inverse of [`pith_math_dct2`],
/// under the same ownership and validation contract.
///
/// # Safety
///
/// See [`pith_math_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_idct2(
    input: *const f64,
    in_len: usize,
    out: *mut *mut f64,
    out_len: *mut usize,
) -> i32 {
    unsafe { run_alloc(input, in_len, out, out_len, idct2_core) }
}

/// Separable 2D orthonormal DCT-II over a `w × h` row-major matrix.
///
/// `in_len` must equal `w·h` and both dimensions must be positive; the
/// direct kernel needs no power-of-two constraint. The output is a
/// fresh buffer of the same `w·h` length, owned by the caller.
///
/// # Safety
///
/// See [`pith_math_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_dct2_2d(
    input: *const f64,
    in_len: usize,
    w: usize,
    h: usize,
    out: *mut *mut f64,
    out_len: *mut usize,
) -> i32 {
    if input.is_null() || out.is_null() || out_len.is_null() {
        return PITH_E_INVALID;
    }
    let x = unsafe { core::slice::from_raw_parts(input, in_len) };
    match dct2_2d_core(x, w, h) {
        Ok(result) => unsafe { hand_out(out, out_len, result) },
        Err(status) => status,
    }
}

/// Forward DFT over `in_len / 2` interleaved complex pairs
/// (`[re, im, …]`), output in the same interleaved layout.
///
/// The complex count must be a non-zero power of two (`in_len` even
/// and `in_len / 2` a power of two) — anything else is
/// [`PITH_E_INVALID`], mirroring the core's panicking precondition.
///
/// # Safety
///
/// See [`pith_math_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_fft(
    input: *const f64,
    in_len: usize,
    out: *mut *mut f64,
    out_len: *mut usize,
) -> i32 {
    unsafe { run_alloc(input, in_len, out, out_len, fft_core) }
}

/// Inverse DFT, the exact (within ulps) inverse of [`pith_math_fft`],
/// under the same interleaved layout and validation contract.
///
/// # Safety
///
/// See [`pith_math_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_ifft(
    input: *const f64,
    in_len: usize,
    out: *mut *mut f64,
    out_len: *mut usize,
) -> i32 {
    unsafe { run_alloc(input, in_len, out, out_len, ifft_core) }
}

/// Forward DFT of a real signal of `in_len` samples; the output is the
/// full `2·in_len` interleaved spectrum (conjugate mirror included).
///
/// `in_len` must be a non-zero power of two.
///
/// # Safety
///
/// See [`pith_math_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_fft_real(
    input: *const f64,
    in_len: usize,
    out: *mut *mut f64,
    out_len: *mut usize,
) -> i32 {
    unsafe { run_alloc(input, in_len, out, out_len, fft_real_core) }
}

/// Solves the 3×3 system `A·x = b` by Gaussian elimination with
/// partial pivoting.
///
/// `input` carries 12 `f64`s: the row-major `A` followed by `b`. On
/// success `out` receives a fresh 3-element (24-byte) buffer. A
/// singular (or scale-relative near-singular, `|pivot| <= ε·s`) system
/// is [`PITH_E_REJECTED`] — ordinary RANSAC data, not a caller bug.
///
/// # Safety
///
/// See [`pith_math_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_solve3(
    input: *const f64,
    in_len: usize,
    out: *mut *mut f64,
    out_len: *mut usize,
) -> i32 {
    unsafe { run_alloc(input, in_len, out, out_len, solve3_core) }
}

/// Determinant of the row-major 3×3 matrix in `input` (9 `f64`s),
/// written through the scalar `out` slot. No allocation.
///
/// # Safety
///
/// `input` must point to 9 readable `f64`s and `out` to one writable
/// `f64`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_det3(input: *const f64, in_len: usize, out: *mut f64) -> i32 {
    if input.is_null() || out.is_null() {
        return PITH_E_INVALID;
    }
    let x = unsafe { core::slice::from_raw_parts(input, in_len) };
    match det3_core(x) {
        Ok(value) => {
            unsafe { *out = value };
            PITH_OK
        }
        Err(status) => status,
    }
}

/// Inverse of the row-major 3×3 matrix in `input` (9 `f64`s), as a
/// fresh 9-element buffer. Singular input is [`PITH_E_REJECTED`].
///
/// # Safety
///
/// See [`pith_math_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_inverse3(
    input: *const f64,
    in_len: usize,
    out: *mut *mut f64,
    out_len: *mut usize,
) -> i32 {
    unsafe { run_alloc(input, in_len, out, out_len, inverse3_core) }
}

/// Transpose of the row-major 3×3 matrix in `input` (9 `f64`s), as a
/// fresh 9-element buffer.
///
/// # Safety
///
/// See [`pith_math_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_transpose3(
    input: *const f64,
    in_len: usize,
    out: *mut *mut f64,
    out_len: *mut usize,
) -> i32 {
    unsafe { run_alloc(input, in_len, out, out_len, transpose3_core) }
}

/// Matrix product `a·b` of the two row-major 3×3 factors packed back
/// to back in `input` (18 `f64`s), as a fresh 9-element buffer.
///
/// # Safety
///
/// See [`pith_math_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_mat3_mul(
    input: *const f64,
    in_len: usize,
    out: *mut *mut f64,
    out_len: *mut usize,
) -> i32 {
    unsafe { run_alloc(input, in_len, out, out_len, mat3_mul_core) }
}

/// Matrix-vector product `m·v` (9 matrix entries then 3 vector entries
/// in `input`), as a fresh 3-element buffer.
///
/// # Safety
///
/// See [`pith_math_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_mat3_mul_vec(
    input: *const f64,
    in_len: usize,
    out: *mut *mut f64,
    out_len: *mut usize,
) -> i32 {
    unsafe { run_alloc(input, in_len, out, out_len, mat3_mul_vec_core) }
}

/// Median of `in_len` `f64`s — the core's lower-middle order statistic
/// (never the mean of the two middles), written through the scalar
/// `out` slot without touching the caller's buffer. An empty input is
/// [`PITH_E_INVALID`].
///
/// # Safety
///
/// `input` must point to `in_len` readable `f64`s and `out` to one
/// writable `f64`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_median(input: *const f64, in_len: usize, out: *mut f64) -> i32 {
    if input.is_null() || out.is_null() {
        return PITH_E_INVALID;
    }
    let x = unsafe { core::slice::from_raw_parts(input, in_len) };
    match median_core(x) {
        Ok(value) => {
            unsafe { *out = value };
            PITH_OK
        }
        Err(status) => status,
    }
}

/// Arbitrary-length forward DFT (Bluestein chirp-z) over `in_len / 2`
/// interleaved complex pairs, output in the same layout.
///
/// `n = in_len / 2` must be at least 1 — **any** length is legal, the
/// power-of-two restriction of [`pith_math_fft`] does not apply here.
///
/// # Safety
///
/// See [`pith_math_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_fft_n(
    input: *const f64,
    in_len: usize,
    out: *mut *mut f64,
    out_len: *mut usize,
) -> i32 {
    unsafe { run_alloc(input, in_len, out, out_len, fft_n_core) }
}

/// 1D orthonormal DCT-III — the first-class forward transform whose
/// inverse is [`pith_math_dct2`], under the same ownership and
/// validation contract as [`pith_math_idct2`].
///
/// # Safety
///
/// See [`pith_math_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_dct3(
    input: *const f64,
    in_len: usize,
    out: *mut *mut f64,
    out_len: *mut usize,
) -> i32 {
    unsafe { run_alloc(input, in_len, out, out_len, dct3_core) }
}

/// 2D orthonormal DCT-III over a `w × h` row-major matrix — the exact
/// inverse of [`pith_math_dct2_2d`], same geometry contract.
///
/// # Safety
///
/// See [`pith_math_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_dct3_2d(
    input: *const f64,
    in_len: usize,
    w: usize,
    h: usize,
    out: *mut *mut f64,
    out_len: *mut usize,
) -> i32 {
    if input.is_null() || out.is_null() || out_len.is_null() {
        return PITH_E_INVALID;
    }
    let x = unsafe { core::slice::from_raw_parts(input, in_len) };
    match dct3_2d_core(x, w, h) {
        Ok(result) => unsafe { hand_out(out, out_len, result) },
        Err(status) => status,
    }
}

/// Inverse of [`pith_math_fft_n`]: any `n ≥ 1`, conjugate-normalized.
///
/// # Safety
///
/// See [`pith_math_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_ifft_n(
    input: *const f64,
    in_len: usize,
    out: *mut *mut f64,
    out_len: *mut usize,
) -> i32 {
    unsafe { run_alloc(input, in_len, out, out_len, ifft_n_core) }
}

/// Full-support linear convolution of the packed operands: `input`
/// carries `a` (`a_len` elements) followed by `b`; the fresh result
/// holds `a_len + b_len − 1` samples. Empty operands and `a_len == 0`
/// are [`PITH_E_INVALID`].
///
/// # Safety
///
/// See [`pith_math_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_conv(
    input: *const f64,
    in_len: usize,
    a_len: usize,
    out: *mut *mut f64,
    out_len: *mut usize,
) -> i32 {
    if let Some(status) = null_status(input, out, out_len) {
        return status;
    }
    let x = unsafe { core::slice::from_raw_parts(input, in_len) };
    match packed_pair_core(x, a_len, convolve) {
        Ok(result) => unsafe { hand_out(out, out_len, result) },
        Err(status) => status,
    }
}

/// Full-support cross-correlation of the packed operands —
/// [`pith_math_conv`] against the reversed second operand — under the
/// same packing and ownership contract.
///
/// # Safety
///
/// See [`pith_math_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_corr(
    input: *const f64,
    in_len: usize,
    a_len: usize,
    out: *mut *mut f64,
    out_len: *mut usize,
) -> i32 {
    if let Some(status) = null_status(input, out, out_len) {
        return status;
    }
    let x = unsafe { core::slice::from_raw_parts(input, in_len) };
    match packed_pair_core(x, a_len, correlate) {
        Ok(result) => unsafe { hand_out(out, out_len, result) },
        Err(status) => status,
    }
}

/// Arithmetic mean of `in_len` `f64`s, through the scalar `out` slot.
/// An empty input is [`PITH_E_INVALID`].
///
/// # Safety
///
/// `input` must point to `in_len` readable `f64`s and `out` to one
/// writable `f64`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_mean(input: *const f64, in_len: usize, out: *mut f64) -> i32 {
    unsafe { run_scalar(input, in_len, out, mean_core) }
}

/// Sample variance (the `n − 1` denominator) of `in_len` `f64`s.
/// Fewer than two observations is [`PITH_E_INVALID`].
///
/// # Safety
///
/// See [`pith_math_mean`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_var(input: *const f64, in_len: usize, out: *mut f64) -> i32 {
    unsafe { run_scalar(input, in_len, out, var_core) }
}

/// Sample covariance of two equal-length series packed back to back
/// in `input` (each half at least two observations), through the
/// scalar `out` slot. A zero-mean-cut layout (`in_len` odd) or short
/// halves is [`PITH_E_INVALID`].
///
/// # Safety
///
/// See [`pith_math_mean`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_cov(input: *const f64, in_len: usize, out: *mut f64) -> i32 {
    unsafe { run_scalar(input, in_len, out, cov_core) }
}

/// Evaluates the Lagrange interpolant through the packed `(x, y)`
/// pairs in `input` at the abscissa `x`, through the scalar `out`
/// slot. Duplicated abscissae are [`PITH_E_REJECTED`].
///
/// # Safety
///
/// `input` must point to `in_len` readable `f64`s (an even count, the
/// interleaved pairs) and `out` to one writable `f64`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_lagrange(
    input: *const f64,
    in_len: usize,
    x: f64,
    out: *mut f64,
) -> i32 {
    if let Some(status) = null_status_scalar(input, out) {
        return status;
    }
    let pts = unsafe { core::slice::from_raw_parts(input, in_len) };
    match lagrange_core(pts, x) {
        Ok(value) => {
            unsafe { *out = value };
            PITH_OK
        }
        Err(status) => status,
    }
}

/// DTW distance between the packed sequences `a` (`a_len` elements)
/// followed by `b`, through the scalar `out` slot. Empty operands are
/// [`PITH_E_INVALID`].
///
/// # Safety
///
/// See [`pith_math_mean`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_dtw(
    input: *const f64,
    in_len: usize,
    a_len: usize,
    out: *mut f64,
) -> i32 {
    if let Some(status) = null_status_scalar(input, out) {
        return status;
    }
    let x = unsafe { core::slice::from_raw_parts(input, in_len) };
    match dtw_core(x, a_len) {
        Ok(value) => {
            unsafe { *out = value };
            PITH_OK
        }
        Err(status) => status,
    }
}

/// Seeded RANSAC line fit over the packed `(x, y)` points in `input`:
/// `iterations` two-point samples under the inlier `threshold`,
/// driven by the `seed`ed SplitMix64 stream. On success the fresh
/// 3-element result carries `[slope, intercept, inlier_count]`.
///
/// Degenerate configuration (`in_len < 4`, odd, zero iterations,
/// `threshold ≤ 0`) is [`PITH_E_INVALID`]; a run that found no model
/// at all is [`PITH_E_REJECTED`]. Seeded runs replay bit-for-bit on
/// every platform.
///
/// # Safety
///
/// See [`pith_math_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_ransac_line(
    input: *const f64,
    in_len: usize,
    threshold: f64,
    iterations: usize,
    seed: u64,
    out: *mut *mut f64,
    out_len: *mut usize,
) -> i32 {
    if let Some(status) = null_status(input, out, out_len) {
        return status;
    }
    let x = unsafe { core::slice::from_raw_parts(input, in_len) };
    match ransac_core(x, threshold, iterations, seed) {
        Ok(result) => unsafe { hand_out(out, out_len, result) },
        Err(status) => status,
    }
}

/// Complex product of the two interleaved pairs in `input`, as a
/// fresh interleaved pair.
///
/// # Safety
///
/// See [`pith_math_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_complex_mul(
    input: *const f64,
    in_len: usize,
    out: *mut *mut f64,
    out_len: *mut usize,
) -> i32 {
    unsafe { run_alloc(input, in_len, out, out_len, complex_mul_core) }
}

/// Complex quotient (Smith's algorithm) of the two interleaved pairs
/// in `input`, as a fresh interleaved pair.
///
/// # Safety
///
/// See [`pith_math_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_complex_div(
    input: *const f64,
    in_len: usize,
    out: *mut *mut f64,
    out_len: *mut usize,
) -> i32 {
    unsafe { run_alloc(input, in_len, out, out_len, complex_div_core) }
}

/// Complex exponential `e^z` of the interleaved pair in `input`.
///
/// # Safety
///
/// See [`pith_math_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_complex_exp(
    input: *const f64,
    in_len: usize,
    out: *mut *mut f64,
    out_len: *mut usize,
) -> i32 {
    unsafe {
        run_alloc(input, in_len, out, out_len, |f| {
            complex_unary_core(f, Complex::exp)
        })
    }
}

/// Principal square root of the interleaved pair in `input`.
///
/// # Safety
///
/// See [`pith_math_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_complex_sqrt(
    input: *const f64,
    in_len: usize,
    out: *mut *mut f64,
    out_len: *mut usize,
) -> i32 {
    unsafe {
        run_alloc(input, in_len, out, out_len, |f| {
            complex_unary_core(f, Complex::sqrt)
        })
    }
}

/// The safe core of [`pith_math_complex_log`]: `z = 0` is
/// [`PITH_E_REJECTED`] (outside the principal-branch domain).
fn complex_log_core(flat: &[f64]) -> Result<Vec<f64>, i32> {
    if flat.len() == 2 && flat[0] == 0.0 && flat[1] == 0.0 {
        return Err(PITH_E_REJECTED);
    }
    complex_unary_core(flat, Complex::ln)
}

/// Principal natural logarithm `[ln|z|, arg z]` of the interleaved
/// pair in `input`; `z = 0` is [`PITH_E_REJECTED`] (domain refusal).
///
/// # Safety
///
/// See [`pith_math_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_complex_log(
    input: *const f64,
    in_len: usize,
    out: *mut *mut f64,
    out_len: *mut usize,
) -> i32 {
    unsafe { run_alloc(input, in_len, out, out_len, complex_log_core) }
}

/// Integer power `z^n` of the interleaved pair in `input`, by
/// squaring, as a fresh interleaved pair.
///
/// # Safety
///
/// See [`pith_math_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_complex_powi(
    input: *const f64,
    in_len: usize,
    n: usize,
    out: *mut *mut f64,
    out_len: *mut usize,
) -> i32 {
    if let Some(status) = null_status(input, out, out_len) {
        return status;
    }
    let x = unsafe { core::slice::from_raw_parts(input, in_len) };
    match complex_powi_core(x, n) {
        Ok(result) => unsafe { hand_out(out, out_len, result) },
        Err(status) => status,
    }
}

/// Principal argument `arg z ∈ (−π, π]` of the interleaved pair in
/// `input`, through the scalar `out` slot.
///
/// # Safety
///
/// See [`pith_math_mean`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_complex_arg(
    input: *const f64,
    in_len: usize,
    out: *mut f64,
) -> i32 {
    unsafe { run_scalar(input, in_len, out, complex_arg_core) }
}

/// Releases a buffer handed out by any allocating export of this
/// module, passing back the same byte count `out_len` reported. Null
/// is accepted and ignored, so callers can free unconditionally on the
/// error path.
///
/// # Safety
///
/// `ptr` must be a pointer the cdylib handed out with the `out_len`
/// byte count that came back with it, and must not have been released
/// (or otherwise freed) before.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_math_free(ptr: *mut f64, len: usize) {
    if ptr.is_null() {
        return;
    }
    let count = len / core::mem::size_of::<f64>();
    let slice = unsafe { core::slice::from_raw_parts_mut(ptr, count) };
    drop(unsafe { Box::from_raw(slice) });
}

#[cfg(test)]
mod tests {
    use super::{
        PITH_E_INVALID, PITH_E_REJECTED, PITH_OK, dct2_core, det3_core, fft_core, fft_real_core,
        flat_complex, ifft_core, inverse3_core, mat3_mul_core, mat3_mul_vec_core, median_core,
        pith_math_dct2, pith_math_dct2_2d, pith_math_det3, pith_math_fft, pith_math_fft_real,
        pith_math_free, pith_math_idct2, pith_math_ifft, pith_math_inverse3, pith_math_mat3_mul,
        pith_math_mat3_mul_vec, pith_math_median, pith_math_solve3, pith_math_transpose3,
        solve3_core, transpose3_core,
    };
    use crate::{dct2, dct2_2d, fft_real, idct2, inverse3, mat3_mul, mat3_mul_vec, transpose3};

    /// Calls an allocating export through raw pointers and, on success,
    /// copies the handed-out buffer back and frees it. `count` is the
    /// expected element count, derived the way every SDK derives it:
    /// from the byte count.
    fn call_alloc(
        op: impl Fn(*const f64, usize, *mut *mut f64, *mut usize) -> i32,
        input: &[f64],
    ) -> (i32, Vec<f64>) {
        let mut out: *mut f64 = core::ptr::null_mut();
        let mut out_len: usize = 0;
        let status = op(input.as_ptr(), input.len(), &mut out, &mut out_len);
        if status != PITH_OK {
            return (status, Vec::new());
        }
        let count = out_len / core::mem::size_of::<f64>();
        let copied = unsafe { core::slice::from_raw_parts(out, count) }.to_vec();
        unsafe { pith_math_free(out, out_len) };
        (status, copied)
    }

    /// Calls a scalar-out export through raw pointers.
    fn call_scalar(op: impl Fn(*const f64, usize, *mut f64) -> i32, input: &[f64]) -> (i32, f64) {
        let mut slot: f64 = 0.0;
        let status = op(input.as_ptr(), input.len(), &mut slot);
        (status, slot)
    }

    /// Every allocating op, driven through the raw FFI with a valid
    /// input, must agree element-for-bit with the safe core behind the
    /// public API — including the byte-count ownership round-trip.
    #[test]
    fn ffi_alloc_ops_match_the_safe_core() {
        let x8 = [0.5, -1.25, 2.0, 3.75, -0.125, 1.5, -2.5, 0.75];
        let c8: Vec<f64> = (0..16).map(|i| (i as f64) * 0.25 - 1.5).collect();
        let m64: Vec<f64> = (0..64).map(|i| ((i % 7) as f64) - 2.0).collect();

        let (status, got) = call_alloc(|i, n, o, l| unsafe { pith_math_dct2(i, n, o, l) }, &x8);
        assert_eq!(status, PITH_OK);
        assert_eq!(got, dct2(&x8));

        let (status, got) = call_alloc(|i, n, o, l| unsafe { pith_math_idct2(i, n, o, l) }, &x8);
        assert_eq!(status, PITH_OK);
        assert_eq!(got, idct2(&x8));

        let (status, got) = call_alloc(
            |i, n, o, l| unsafe { pith_math_dct2_2d(i, n, 8, 8, o, l) },
            &m64,
        );
        assert_eq!(status, PITH_OK);
        let mut want = m64.clone();
        dct2_2d(&mut want, 8, 8);
        assert_eq!(got, want);

        let (status, got) = call_alloc(|i, n, o, l| unsafe { pith_math_fft(i, n, o, l) }, &c8);
        assert_eq!(status, PITH_OK);
        assert_eq!(got.len(), c8.len());

        let (status, got) = call_alloc(|i, n, o, l| unsafe { pith_math_ifft(i, n, o, l) }, &c8);
        assert_eq!(status, PITH_OK);
        assert_eq!(got.len(), c8.len());

        let (status, got) = call_alloc(|i, n, o, l| unsafe { pith_math_fft_real(i, n, o, l) }, &x8);
        assert_eq!(status, PITH_OK);
        assert_eq!(got, flat_complex(&fft_real(&x8)));

        let system: Vec<f64> = [2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0]
            .into_iter()
            .chain([8.0, -11.0, -3.0])
            .collect();
        let (status, got) = call_alloc(
            |i, n, o, l| unsafe { pith_math_solve3(i, n, o, l) },
            &system,
        );
        assert_eq!(status, PITH_OK);
        assert_eq!(got.len(), 3);

        let inv = [2.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 2.0];
        let m = super::mat3(&inv);
        let (status, got) =
            call_alloc(|i, n, o, l| unsafe { pith_math_inverse3(i, n, o, l) }, &inv);
        assert_eq!(status, PITH_OK);
        assert_eq!(got, super::mat3_flat(&inverse3(&m).unwrap()));

        let (status, got) = call_alloc(
            |i, n, o, l| unsafe { pith_math_transpose3(i, n, o, l) },
            &inv,
        );
        assert_eq!(status, PITH_OK);
        assert_eq!(got, super::mat3_flat(&transpose3(&m)));

        let ab: Vec<f64> = inv.iter().copied().chain(inv.iter().copied()).collect();
        let (status, got) = call_alloc(|i, n, o, l| unsafe { pith_math_mat3_mul(i, n, o, l) }, &ab);
        assert_eq!(status, PITH_OK);
        assert_eq!(got, super::mat3_flat(&mat3_mul(&m, &m)));

        let mv: Vec<f64> = inv.iter().copied().chain([1.0, 2.0, 3.0]).collect();
        let (status, got) = call_alloc(
            |i, n, o, l| unsafe { pith_math_mat3_mul_vec(i, n, o, l) },
            &mv,
        );
        assert_eq!(status, PITH_OK);
        assert_eq!(got, mat3_mul_vec(&m, &[1.0, 2.0, 3.0]).to_vec());
    }

    /// det3.standard, pinned literally from `tests/reference.json` —
    /// input bits, output bit — read back through the scalar out-slot.
    #[test]
    fn det3_standard_pins_the_reference_bits() {
        let input: Vec<f64> = [
            "4018000000000000",
            "3ff0000000000000",
            "3ff0000000000000",
            "4010000000000000",
            "c000000000000000",
            "4014000000000000",
            "4000000000000000",
            "4020000000000000",
            "401c000000000000",
        ]
        .iter()
        .map(|h| f64::from_bits(u64::from_str_radix(h, 16).unwrap()))
        .collect();
        let (status, got) = call_scalar(|i, n, o| unsafe { pith_math_det3(i, n, o) }, &input);
        assert_eq!(status, PITH_OK);
        assert_eq!(
            got.to_bits(),
            u64::from_str_radix("c073200000000000", 16).unwrap()
        );

        // The safe core agrees.
        assert_eq!(det3_core(&input), Ok(f64::from_bits(0xc073200000000000)));
    }

    /// One approx vector (fft.n8's first four bins) must land within
    /// the recorded budgets — max(tol_abs, tol_rel·|expected|) — for
    /// at least the first four bins, each value pinned literally from
    /// `tests/reference.json` (the rust-derived pins the SDKs mirror).
    #[test]
    fn fft_n8_first_bins_land_within_the_recorded_tolerance() {
        let input: Vec<f64> = [
            "3ff0000000000000",
            "3fe0000000000000",
            "bfe0000000000000",
            "4000000000000000",
            "3fd0000000000000",
            "bff8000000000000",
            "3ffc000000000000",
            "0000000000000000",
            "c002000000000000",
            "3fe0000000000000",
            "3fc0000000000000",
            "4008000000000000",
            "bfe8000000000000",
            "3ff0000000000000",
            "4004000000000000",
            "bff0000000000000",
        ]
        .iter()
        .map(|h| f64::from_bits(u64::from_str_radix(h, 16).unwrap()))
        .collect();
        let want: Vec<f64> = [
            "4001000000000000",
            "4012000000000000",
            "3fead413cccfe77a",
            "bff712318007c2b1",
        ]
        .iter()
        .map(|h| f64::from_bits(u64::from_str_radix(h, 16).unwrap()))
        .collect();
        let tol_abs = f64::from_bits(u64::from_str_radix("3d3c25c268497682", 16).unwrap());
        let tol_rel = f64::from_bits(u64::from_str_radix("3d719799812dea11", 16).unwrap());

        let got = fft_core(&input).expect("fft.n8");
        for (g, &w) in got.iter().take(4).zip(&want) {
            assert!((g - w).abs() <= tol_abs.max(tol_rel * w.abs()));
        }
    }

    /// The scalar-out exports: median is the core's lower-middle
    /// convention (median_copy), NaN sorts last and never crashes.
    #[test]
    fn ffi_scalar_ops_match_the_safe_core() {
        let (status, got) = call_scalar(
            |i, n, o| unsafe { pith_math_median(i, n, o) },
            &[1.0, 2.0, 3.0, 4.0],
        );
        assert_eq!((status, got), (PITH_OK, 2.0));

        let nan = f64::from_bits(0x7ff8000000000000);
        let (status, got) = call_scalar(
            |i, n, o| unsafe { pith_math_median(i, n, o) },
            &[1.0, nan, 2.0, 3.0],
        );
        assert_eq!((status, got), (PITH_OK, 2.0));
        assert_eq!(got.to_bits(), 0x4000000000000000);

        assert_eq!(median_core(&[]), Err(PITH_E_INVALID));
    }

    /// Every refusal path returns a status code, never a panic: null
    /// pointers, broken geometry, empty inputs, singular systems.
    #[test]
    fn ffi_refusals() {
        let mut out: *mut f64 = core::ptr::null_mut();
        let mut out_len: usize = 0;
        let mut slot: f64 = 0.0;

        // Null data pointer, every signature shape.
        assert_eq!(
            unsafe { pith_math_dct2(core::ptr::null(), 0, &mut out, &mut out_len) },
            PITH_E_INVALID
        );
        assert_eq!(
            unsafe { pith_math_dct2_2d(core::ptr::null(), 0, 1, 1, &mut out, &mut out_len) },
            PITH_E_INVALID
        );
        assert_eq!(
            unsafe { pith_math_det3(core::ptr::null(), 0, &mut slot) },
            PITH_E_INVALID
        );
        assert_eq!(
            unsafe { pith_math_median(core::ptr::null(), 0, &mut slot) },
            PITH_E_INVALID
        );

        let x8 = [0.5, -1.25, 2.0, 3.75, -0.125, 1.5, -2.5, 0.75];

        // Null out / out_len slots.
        assert_eq!(
            unsafe { pith_math_dct2(x8.as_ptr(), x8.len(), core::ptr::null_mut(), &mut out_len) },
            PITH_E_INVALID
        );
        assert_eq!(
            unsafe { pith_math_dct2(x8.as_ptr(), x8.len(), &mut out, core::ptr::null_mut()) },
            PITH_E_INVALID
        );
        assert_eq!(
            unsafe { pith_math_det3(x8.as_ptr(), 8, core::ptr::null_mut()) },
            PITH_E_INVALID
        );

        // fft/ifft: odd element count (6 = 3 pairs, 3 not a power of
        // two) and zero pairs.
        assert_eq!(
            call_alloc(|i, n, o, l| unsafe { pith_math_fft(i, n, o, l) }, &[0.0; 6]).0,
            PITH_E_INVALID
        );
        assert_eq!(
            call_alloc(|i, n, o, l| unsafe { pith_math_fft(i, n, o, l) }, &[]).0,
            PITH_E_INVALID
        );
        assert_eq!(
            call_alloc(|i, n, o, l| unsafe { pith_math_fft(i, n, o, l) }, &[0.0; 7]).0,
            PITH_E_INVALID
        );
        assert_eq!(
            call_alloc(
                |i, n, o, l| unsafe { pith_math_ifft(i, n, o, l) },
                &[0.0; 6]
            )
            .0,
            PITH_E_INVALID
        );

        // fft_real: non-power-of-two and empty.
        assert_eq!(
            call_alloc(
                |i, n, o, l| unsafe { pith_math_fft_real(i, n, o, l) },
                &[0.0; 3]
            )
            .0,
            PITH_E_INVALID
        );
        assert_eq!(
            call_alloc(|i, n, o, l| unsafe { pith_math_fft_real(i, n, o, l) }, &[]).0,
            PITH_E_INVALID
        );

        // Empty inputs.
        assert_eq!(
            call_alloc(|i, n, o, l| unsafe { pith_math_dct2(i, n, o, l) }, &[]).0,
            PITH_E_INVALID
        );
        assert_eq!(
            call_alloc(|i, n, o, l| unsafe { pith_math_idct2(i, n, o, l) }, &[]).0,
            PITH_E_INVALID
        );

        // dct2_2d geometry.
        assert_eq!(
            call_alloc(
                |i, n, o, l| unsafe { pith_math_dct2_2d(i, n, 4, 8, o, l) },
                &x8
            )
            .0,
            PITH_E_INVALID
        );
        assert_eq!(
            call_alloc(
                |i, n, o, l| unsafe { pith_math_dct2_2d(i, n, 0, 8, o, l) },
                &x8
            )
            .0,
            PITH_E_INVALID
        );

        // Length mismatches.
        assert_eq!(
            call_alloc(
                |i, n, o, l| unsafe { pith_math_solve3(i, n, o, l) },
                &[0.0; 11]
            )
            .0,
            PITH_E_INVALID
        );
        assert_eq!(
            call_scalar(|i, n, o| unsafe { pith_math_det3(i, n, o) }, &[0.0; 8]).0,
            PITH_E_INVALID
        );
        assert_eq!(
            call_alloc(
                |i, n, o, l| unsafe { pith_math_inverse3(i, n, o, l) },
                &[0.0; 8]
            )
            .0,
            PITH_E_INVALID
        );
        assert_eq!(
            call_alloc(
                |i, n, o, l| unsafe { pith_math_transpose3(i, n, o, l) },
                &[0.0; 8]
            )
            .0,
            PITH_E_INVALID
        );
        assert_eq!(
            call_alloc(
                |i, n, o, l| unsafe { pith_math_mat3_mul(i, n, o, l) },
                &[0.0; 17]
            )
            .0,
            PITH_E_INVALID
        );
        assert_eq!(
            call_alloc(
                |i, n, o, l| unsafe { pith_math_mat3_mul_vec(i, n, o, l) },
                &[0.0; 13]
            )
            .0,
            PITH_E_INVALID
        );

        // Singular systems are rejected, not crashed on.
        assert_eq!(
            call_alloc(
                |i, n, o, l| unsafe { pith_math_solve3(i, n, o, l) },
                &[0.0; 12]
            )
            .0,
            PITH_E_REJECTED
        );
        assert_eq!(
            call_alloc(
                |i, n, o, l| unsafe { pith_math_inverse3(i, n, o, l) },
                &[0.0; 9]
            )
            .0,
            PITH_E_REJECTED
        );

        // A null buffer is a legal free.
        unsafe { pith_math_free(core::ptr::null_mut(), 0) };
    }

    /// The safe cores behind every op, exercised without raw pointers
    /// (these are the lines the coverage gate measures the SDK
    /// refusals against).
    #[test]
    fn safe_cores_reject_and_compose() {
        let x8 = [0.5, -1.25, 2.0, 3.75, -0.125, 1.5, -2.5, 0.75];
        assert_eq!(dct2_core(&x8).unwrap(), dct2(&x8));
        assert_eq!(dct2_core(&[]), Err(PITH_E_INVALID));
        assert_eq!(ifft_core(&[0.0; 6]), Err(PITH_E_INVALID));
        assert_eq!(fft_real_core(&[0.0; 3]), Err(PITH_E_INVALID));

        // idct2(dct2(x)) round-trips through the cores.
        let fwd = dct2_core(&x8).unwrap();
        let back = super::idct2_core(&fwd).unwrap();
        for (b, &x) in back.iter().zip(&x8) {
            assert!((b - x).abs() < 1e-12);
        }

        assert_eq!(mat3_mul_core(&[0.0; 18]).unwrap().len(), 9);
        assert_eq!(mat3_mul_vec_core(&[0.0; 12]).unwrap().len(), 3);
        assert_eq!(transpose3_core(&[0.0; 9]).unwrap().len(), 9);
        assert_eq!(inverse3_core(&[0.0; 9]), Err(PITH_E_REJECTED));
        assert_eq!(solve3_core(&[0.0; 12]), Err(PITH_E_REJECTED));
        assert_eq!(det3_core(&[0.0; 9]), Ok(0.0));
    }
}
