//! The JNI surface of `pith-math`: the entry points the Java SDK binds
//! through `System.loadLibrary("pith_math")`.
//!
//! The suite's Java FFI convention, defined by this module:
//!
//! * one flat set of `#[unsafe(no_mangle)] pub unsafe extern "system"`
//!   functions named `Java_pith_math_PithMath_<op>` for the static
//!   natives of `pith.math.PithMath` — no overloads, so no mangled
//!   signature suffixes;
//! * arrays cross as `jdoubleArray` in and out; scalars as `jdouble`
//!   returns; geometry as `jint`/`jlong` parameters — no structs;
//! * **an allocating op returns a fresh, GC-owned `jdoubleArray`** —
//!   there is no `pith_math_free` on this surface, the Java garbage
//!   collector owns every result;
//! * an error is a **`NULL` array** or a **`NaN` scalar**: the Java
//!   wrapper pre-validates every caller bug (null arrays, empty
//!   inputs, length/geometry mismatches, non-positive iterations) and
//!   throws before crossing the boundary, so a crossing that fails is
//!   a *data* refusal (a singular matrix, duplicated Lagrange nodes,
//!   an all-degenerate RANSAC run) and surfaces in Java as
//!   `FfiError`("rejected");
//! * the JNI function table is addressed **by index**, not by a
//!   hand-written 233-field struct: every slot is pointer-sized, so
//!   `(*env)[IDX]` transmutes straight to the typed entry point. The
//!   indices used are the JNI spec constants below;
//! * the `unsafe` allowance is confined to this module, like
//!   [`crate::ffi`]; every core module stays unsafe-free.
//!
//! The fake-env tests (`tests/ffi_jni_env.rs`) build a synthetic
//! function table with the same indices and drive every export
//! through it — the JNI surface is covered without a JVM.

#![allow(unsafe_code)]

use crate::bluestein::{fft_arbitrary, ifft_arbitrary as core_ifft_n};
use crate::complex::Complex;
use crate::conv::{convolve, correlate};
use crate::dct::{dct2 as core_dct2, dct2_2d as core_dct2_2d, idct2};
use crate::dct3::{dct3 as core_dct3, dct3_2d as core_dct3_2d};
use crate::dtw::dtw_distance;
use crate::fft::{fft as core_fft, fft_real, ifft};
use crate::interp::lagrange_eval;
use crate::linalg::{det3, inverse3, mat3_mul, mat3_mul_vec, transpose3};
use crate::median::median_copy;
use crate::ransac::ransac_line;
use crate::solve3::solve3;
use crate::stats::{covariance, mean, variance};

// JNI function-table indices (the <jni.h> order, 0-based).
const IDX_GET_ARRAY_LENGTH: usize = 171;
const IDX_NEW_DOUBLE_ARRAY: usize = 182;
const IDX_GET_DOUBLE_ARRAY_ELEMENTS: usize = 190;
const IDX_RELEASE_DOUBLE_ARRAY_ELEMENTS: usize = 198;
const IDX_SET_DOUBLE_ARRAY_REGION: usize = 214;

/// The env is a pointer to the function table; slots are all
/// pointer-sized, so indexing `[usize; N]` matches the spec layout.
type RawEnv = *const *const [usize; 256];

/// `jdoubleArray`, `jclass`, `jobject` are opaque handles across the
/// boundary; the bindings never dereference them.
#[allow(clippy::upper_case_acronyms)]
type JObject = *mut core::ffi::c_void;
#[allow(clippy::upper_case_acronyms)]
type JDoubleArray = JObject;
#[allow(clippy::upper_case_acronyms)]
type JClass = JObject;

/// Reads a Java double array into a Rust `Vec` (elements + length)
/// through the table, releasing the JVM's copy immediately.
///
/// # Safety
///
/// `env` must be a live JNIEnv and `arr` a live `jdoubleArray` valid
/// for the duration of the call.
unsafe fn read_array(env: RawEnv, arr: JDoubleArray) -> Vec<f64> {
    let table: &[usize; 256] = unsafe { &**env };
    let len = unsafe {
        let f: extern "system" fn(RawEnv, JObject, *mut u8) -> i32 =
            core::mem::transmute(table[IDX_GET_ARRAY_LENGTH]);
        f(env, arr, core::ptr::null_mut()) as usize
    };
    let mut out = vec![0.0f64; len];
    unsafe {
        let f: extern "system" fn(RawEnv, JDoubleArray, *mut f64, *mut u8) -> *mut f64 =
            core::mem::transmute(table[IDX_GET_DOUBLE_ARRAY_ELEMENTS]);
        let data = f(env, arr, core::ptr::null_mut(), core::ptr::null_mut());
        out.copy_from_slice(core::slice::from_raw_parts(data, len));
        let r: extern "system" fn(RawEnv, JDoubleArray, *mut f64, i32) =
            core::mem::transmute(table[IDX_RELEASE_DOUBLE_ARRAY_ELEMENTS]);
        r(env, arr, data, 0);
    }
    out
}

/// Writes a Rust slice into a freshly allocated, GC-owned
/// `jdoubleArray`.
///
/// # Safety
///
/// `env` must be a live JNIEnv.
unsafe fn write_array(env: RawEnv, data: &[f64]) -> JDoubleArray {
    let table: &[usize; 256] = unsafe { &**env };
    unsafe {
        let new: extern "system" fn(RawEnv, i32) -> JDoubleArray =
            core::mem::transmute(table[IDX_NEW_DOUBLE_ARRAY]);
        let arr = new(env, data.len() as i32);
        if arr.is_null() {
            return core::ptr::null_mut();
        }
        let set: extern "system" fn(RawEnv, JDoubleArray, i32, i32, *const f64) =
            core::mem::transmute(table[IDX_SET_DOUBLE_ARRAY_REGION]);
        set(env, arr, 0, data.len() as i32, data.as_ptr());
        arr
    }
}

/// The error return for an allocating op: a NULL array. The Java
/// wrapper translates it to `FfiError("rejected")` — it has already
/// thrown at every caller bug before crossing.
fn null_array() -> JDoubleArray {
    core::ptr::null_mut()
}

/// The error return for a scalar op: `NaN` (unreachable for every
/// valid input of the wrapped kernels).
fn nan() -> f64 {
    f64::NAN
}

/// Packs two Java arrays as the flat operand pair the cores take.
///
/// # Safety
///
/// `env` must be live; both arrays valid.
unsafe fn pair(env: RawEnv, a: JDoubleArray, b: JDoubleArray) -> Result<(Vec<f64>, Vec<f64>), ()> {
    if a.is_null() || b.is_null() {
        return Err(());
    }
    Ok((unsafe { read_array(env, a) }, unsafe { read_array(env, b) }))
}

/// Runs a two-array → allocating kernel; `NULL` arrays are the caller
/// bug the Java side must have thrown at, mirrored defensively as
/// `NULL`. An empty kernel result — the kernels' own refusal shape —
/// surfaces as `NULL` too.
///
/// # Safety
///
/// `env` must be live; both arrays valid.
unsafe fn run_pair_alloc(
    env: RawEnv,
    a: JDoubleArray,
    b: JDoubleArray,
    kernel: impl Fn(&[f64], &[f64]) -> Vec<f64>,
) -> JDoubleArray {
    let Ok((x, y)) = (unsafe { pair(env, a, b) }) else {
        return null_array();
    };
    if x.is_empty() || y.is_empty() {
        return null_array();
    }
    let out = kernel(&x, &y);
    if out.is_empty() {
        return null_array();
    }
    unsafe { write_array(env, &out) }
}

/// Runs a two-array → scalar kernel.
///
/// # Safety
///
/// `env` must be live; both arrays valid.
unsafe fn run_pair_scalar(
    env: RawEnv,
    a: JDoubleArray,
    b: JDoubleArray,
    kernel: impl Fn(&[f64], &[f64]) -> Option<f64>,
) -> f64 {
    let Ok((x, y)) = (unsafe { pair(env, a, b) }) else {
        return nan();
    };
    kernel(&x, &y).unwrap_or_else(nan)
}

/// Reads one array and runs a flat → allocating kernel.
///
/// # Safety
///
/// `env` must be live; the array valid.
unsafe fn run_flat_alloc(
    env: RawEnv,
    arr: JDoubleArray,
    kernel: impl Fn(&[f64]) -> Vec<f64>,
) -> JDoubleArray {
    if arr.is_null() {
        return null_array();
    }
    let x = unsafe { read_array(env, arr) };
    if x.is_empty() {
        return null_array();
    }
    let out = kernel(&x);
    if out.is_empty() {
        return null_array();
    }
    unsafe { write_array(env, &out) }
}

/// Reads one array and runs a flat → scalar kernel.
///
/// # Safety
///
/// `env` must be live; the array valid.
unsafe fn run_flat_scalar(
    env: RawEnv,
    arr: JDoubleArray,
    kernel: impl Fn(&[f64]) -> Option<f64>,
) -> f64 {
    if arr.is_null() {
        return nan();
    }
    let x = unsafe { read_array(env, arr) };
    kernel(&x).unwrap_or_else(nan)
}

/// Interleaves a Java complex pair array into `Complex` values.
fn to_complex(flat: &[f64]) -> Vec<Complex> {
    flat.chunks_exact(2)
        .map(|c| Complex::new(c[0], c[1]))
        .collect()
}

/// Flattens `Complex` values to the interleaved wire form.
fn from_complex(cs: &[Complex]) -> Vec<f64> {
    cs.iter().flat_map(|c| [c.re, c.im]).collect()
}

// -- DCT ----------------------------------------------------------------

/// Java: `PithMath.dct2(double[]) → double[]`.
///
/// # Safety
///
/// `env` must be a live JNIEnv; `x` a live array.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_dct2(
    env: RawEnv,
    _class: JClass,
    x: JDoubleArray,
) -> JDoubleArray {
    unsafe { run_flat_alloc(env, x, core_dct2) }
}

/// Java: `PithMath.idct2(double[]) → double[]` — the orthonormal
/// DCT-III.
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_idct2(
    env: RawEnv,
    _class: JClass,
    x: JDoubleArray,
) -> JDoubleArray {
    unsafe { run_flat_alloc(env, x, idct2) }
}

/// Java: `PithMath.dct3(double[]) → double[]` — the first-class
/// forward DCT-III.
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_dct3(
    env: RawEnv,
    _class: JClass,
    x: JDoubleArray,
) -> JDoubleArray {
    unsafe { run_flat_alloc(env, x, core_dct3) }
}

/// Java: `PithMath.dct2_2d(double[], int w, int h) → double[]`.
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_dct22d(
    env: RawEnv,
    _class: JClass,
    data: JDoubleArray,
    w: i32,
    h: i32,
) -> JDoubleArray {
    if data.is_null() || w <= 0 || h <= 0 {
        return null_array();
    }
    let flat = unsafe { read_array(env, data) };
    if flat.len() != w as usize * h as usize {
        return null_array();
    }
    let mut buf = flat;
    core_dct2_2d(&mut buf, w as usize, h as usize);
    unsafe { write_array(env, &buf) }
}

// -- FFT ----------------------------------------------------------------

/// Java: `PithMath.fft(double[]) → double[]` (radix-2; the complex
/// count must be a power of two — pre-validated in Java).
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_fft(
    env: RawEnv,
    _class: JClass,
    x: JDoubleArray,
) -> JDoubleArray {
    let flat_kernel = |v: &[f64]| -> Vec<f64> {
        let mut buf = to_complex(v);
        if v.len() % 2 != 0 || !buf.len().is_power_of_two() || buf.is_empty() {
            return Vec::new();
        }
        core_fft(&mut buf);
        from_complex(&buf)
    };
    unsafe { run_flat_alloc(env, x, flat_kernel) }
}

/// Java: `PithMath.ifft(double[]) → double[]`.
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_ifft(
    env: RawEnv,
    _class: JClass,
    x: JDoubleArray,
) -> JDoubleArray {
    let flat_kernel = |v: &[f64]| -> Vec<f64> {
        let mut buf = to_complex(v);
        if v.len() % 2 != 0 || !buf.len().is_power_of_two() || buf.is_empty() {
            return Vec::new();
        }
        ifft(&mut buf);
        from_complex(&buf)
    };
    unsafe { run_flat_alloc(env, x, flat_kernel) }
}

/// Java: `PithMath.fftReal(double[]) → double[]` (power-of-two real
/// length, pre-validated in Java).
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_fftReal(
    env: RawEnv,
    _class: JClass,
    x: JDoubleArray,
) -> JDoubleArray {
    let flat_kernel = |v: &[f64]| -> Vec<f64> {
        if !v.len().is_power_of_two() {
            return Vec::new();
        }
        from_complex(&fft_real(v))
    };
    unsafe { run_flat_alloc(env, x, flat_kernel) }
}

/// Java: `PithMath.fftN(double[]) → double[]` — the arbitrary-length
/// (Bluestein) forward DFT over interleaved pairs.
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_fftN(
    env: RawEnv,
    _class: JClass,
    x: JDoubleArray,
) -> JDoubleArray {
    let flat_kernel = |v: &[f64]| -> Vec<f64> {
        if v.len() % 2 != 0 || v.is_empty() {
            return Vec::new();
        }
        let mut buf = to_complex(v);
        fft_arbitrary(&mut buf);
        from_complex(&buf)
    };
    unsafe { run_flat_alloc(env, x, flat_kernel) }
}

// -- 3×3 linalg -----------------------------------------------------------

/// Java: `PithMath.ifftN(double[]) → double[]` — the arbitrary-length
/// inverse.
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_ifftN(
    env: RawEnv,
    _class: JClass,
    x: JDoubleArray,
) -> JDoubleArray {
    let flat_kernel = |v: &[f64]| -> Vec<f64> {
        if v.len() % 2 != 0 || v.is_empty() {
            return Vec::new();
        }
        let mut buf = to_complex(v);
        core_ifft_n(&mut buf);
        from_complex(&buf)
    };
    unsafe { run_flat_alloc(env, x, flat_kernel) }
}

/// Java: `PithMath.solve3(double[] a, double[] b) → double[]`; a
/// singular system is a `NULL` return.
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_solve3(
    env: RawEnv,
    _class: JClass,
    a: JDoubleArray,
    b: JDoubleArray,
) -> JDoubleArray {
    let Ok((m, rhs)) = (unsafe { pair(env, a, b) }) else {
        return null_array();
    };
    if m.len() != 9 || rhs.len() != 3 {
        return null_array();
    }
    let matrix: [[f64; 3]; 3] = core::array::from_fn(|r| core::array::from_fn(|c| m[r * 3 + c]));
    match solve3(&matrix, &[rhs[0], rhs[1], rhs[2]]) {
        Some(x) => unsafe { write_array(env, &x) },
        None => null_array(),
    }
}

/// Java: `PithMath.det3(double[]) → double`.
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_det3(
    env: RawEnv,
    _class: JClass,
    m: JDoubleArray,
) -> f64 {
    unsafe { run_flat_scalar(env, m, |v| (v.len() == 9).then(|| det3(&matrix3(v)))) }
}

/// Java: `PithMath.inverse3(double[]) → double[]`; singular is `NULL`.
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_inverse3(
    env: RawEnv,
    _class: JClass,
    m: JDoubleArray,
) -> JDoubleArray {
    unsafe {
        run_flat_alloc(env, m, |v| {
            if v.len() != 9 {
                return Vec::new();
            }
            inverse3(&matrix3(v)).map(|m| flat3(&m)).unwrap_or_default()
        })
    }
}

/// Java: `PithMath.transpose3(double[]) → double[]`.
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_transpose3(
    env: RawEnv,
    _class: JClass,
    m: JDoubleArray,
) -> JDoubleArray {
    let kernel = |v: &[f64]| -> Vec<f64> {
        if v.len() != 9 {
            return Vec::new();
        }
        flat3(&transpose3(&matrix3(v)))
    };
    unsafe { run_flat_alloc(env, m, kernel) }
}

/// Java: `PithMath.mat3Mul(double[] a, double[] b) → double[]`.
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_mat3Mul(
    env: RawEnv,
    _class: JClass,
    a: JDoubleArray,
    b: JDoubleArray,
) -> JDoubleArray {
    let kernel = |x: &[f64], y: &[f64]| -> Vec<f64> {
        if x.len() != 9 || y.len() != 9 {
            return Vec::new();
        }
        flat3(&mat3_mul(&matrix3(x), &matrix3(y)))
    };
    unsafe { run_pair_alloc(env, a, b, kernel) }
}

/// Java: `PithMath.mat3MulVec(double[] m, double[] v) → double[]`.
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_mat3MulVec(
    env: RawEnv,
    _class: JClass,
    m: JDoubleArray,
    v: JDoubleArray,
) -> JDoubleArray {
    let kernel = |x: &[f64], y: &[f64]| -> Vec<f64> {
        if x.len() != 9 || y.len() != 3 {
            return Vec::new();
        }
        mat3_mul_vec(&matrix3(x), &[y[0], y[1], y[2]]).to_vec()
    };
    unsafe { run_pair_alloc(env, m, v, kernel) }
}

/// Java: `PithMath.median(double[]) → double`.
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_median(
    env: RawEnv,
    _class: JClass,
    x: JDoubleArray,
) -> f64 {
    unsafe { run_flat_scalar(env, x, median_copy) }
}

// -- Tier-1 ops -----------------------------------------------------------

/// Java: `PithMath.conv(double[] a, double[] b) → double[]`.
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_conv(
    env: RawEnv,
    _class: JClass,
    a: JDoubleArray,
    b: JDoubleArray,
) -> JDoubleArray {
    unsafe { run_pair_alloc(env, a, b, convolve) }
}

/// Java: `PithMath.corr(double[] a, double[] b) → double[]`.
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_corr(
    env: RawEnv,
    _class: JClass,
    a: JDoubleArray,
    b: JDoubleArray,
) -> JDoubleArray {
    unsafe { run_pair_alloc(env, a, b, correlate) }
}

/// Java: `PithMath.mean(double[]) → double`.
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_mean(
    env: RawEnv,
    _class: JClass,
    x: JDoubleArray,
) -> f64 {
    unsafe { run_flat_scalar(env, x, mean) }
}

/// Java: `PithMath.variance(double[]) → double` (sample form).
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_variance(
    env: RawEnv,
    _class: JClass,
    x: JDoubleArray,
) -> f64 {
    unsafe { run_flat_scalar(env, x, variance) }
}

/// Java: `PithMath.covariance(double[] a, double[] b) → double`
/// (sample form; length mismatches are a Java-side pre-validation).
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_covariance(
    env: RawEnv,
    _class: JClass,
    a: JDoubleArray,
    b: JDoubleArray,
) -> f64 {
    unsafe { run_pair_scalar(env, a, b, covariance) }
}

/// Java: `PithMath.lagrange(double[] xs, double[] ys, double x) →
/// double`; duplicated nodes are a `NaN` return.
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_lagrange(
    env: RawEnv,
    _class: JClass,
    xs: JDoubleArray,
    ys: JDoubleArray,
    x: f64,
) -> f64 {
    let kernel = |ab: &[f64], ord: &[f64]| -> Option<f64> {
        if ab.len() != ord.len() {
            return None;
        }
        let points: Vec<(f64, f64)> = ab.iter().copied().zip(ord.iter().copied()).collect();
        lagrange_eval(&points, x)
    };
    unsafe { run_pair_scalar(env, xs, ys, kernel) }
}

/// Java: `PithMath.dtw(double[] a, double[] b) → double`.
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_dtw(
    env: RawEnv,
    _class: JClass,
    a: JDoubleArray,
    b: JDoubleArray,
) -> f64 {
    unsafe { run_pair_scalar(env, a, b, dtw_distance) }
}

/// Java: `PithMath.ransacLine(double[] xs, double[] ys, double
/// threshold, int iterations, long seed) → double[]` — a fresh
/// `[slope, intercept, inlierCount]`, or `NULL` when no model was
/// found.
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_ransacLine(
    env: RawEnv,
    _class: JClass,
    xs: JDoubleArray,
    ys: JDoubleArray,
    threshold: f64,
    iterations: i32,
    seed: i64,
) -> JDoubleArray {
    let kernel = |ab: &[f64], ord: &[f64]| -> Vec<f64> {
        if ab.len() != ord.len()
            || ab.len() < 2
            || iterations <= 0
            || threshold.is_nan()
            || threshold <= 0.0
        {
            return Vec::new();
        }
        let points: Vec<(f64, f64)> = ab.iter().copied().zip(ord.iter().copied()).collect();
        ransac_line(&points, threshold, iterations as usize, seed as u64)
            .map(|fit| vec![fit.slope, fit.intercept, fit.inliers as f64])
            .unwrap_or_default()
    };
    unsafe { run_pair_alloc(env, xs, ys, kernel) }
}

// -- Complex arithmetic ---------------------------------------------------

/// Java: `PithMath.complexMul(double[] z, double[] w) → double[]`.
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_complexMul(
    env: RawEnv,
    _class: JClass,
    z: JDoubleArray,
    w: JDoubleArray,
) -> JDoubleArray {
    let kernel = |x: &[f64], y: &[f64]| -> Vec<f64> {
        if x.len() != 2 || y.len() != 2 {
            return Vec::new();
        }
        let (cz, cw) = (Complex::new(x[0], x[1]), Complex::new(y[0], y[1]));
        let r = cz * cw;
        vec![r.re, r.im]
    };
    unsafe { run_pair_alloc(env, z, w, kernel) }
}

/// Java: `PithMath.complexDiv(double[] z, double[] w) → double[]`.
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_complexDiv(
    env: RawEnv,
    _class: JClass,
    z: JDoubleArray,
    w: JDoubleArray,
) -> JDoubleArray {
    let kernel = |x: &[f64], y: &[f64]| -> Vec<f64> {
        if x.len() != 2 || y.len() != 2 || (y[0] == 0.0 && y[1] == 0.0) {
            // A zero denominator is the domain refusal, like a
            // singular matrix: null, not NaN.
            return Vec::new();
        }
        let r = Complex::new(x[0], x[1]) / Complex::new(y[0], y[1]);
        vec![r.re, r.im]
    };
    unsafe { run_pair_alloc(env, z, w, kernel) }
}

/// Java: `PithMath.dct3_2d(double[] data, int w, int h) → double[]` —
/// the exact inverse of `dct2_2d`.
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_dct32d(
    env: RawEnv,
    _class: JClass,
    data: JDoubleArray,
    w: i32,
    h: i32,
) -> JDoubleArray {
    if data.is_null() || w <= 0 || h <= 0 {
        return null_array();
    }
    let flat = unsafe { read_array(env, data) };
    if flat.len() != w as usize * h as usize {
        return null_array();
    }
    let mut buf = flat;
    core_dct3_2d(&mut buf, w as usize, h as usize);
    unsafe { write_array(env, &buf) }
}

/// Java: `PithMath.complexExp(double[] z) → double[]`.
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_complexExp(
    env: RawEnv,
    _class: JClass,
    z: JDoubleArray,
) -> JDoubleArray {
    let kernel = |x: &[f64]| -> Vec<f64> {
        if x.len() != 2 {
            return Vec::new();
        }
        let r = Complex::new(x[0], x[1]).exp();
        vec![r.re, r.im]
    };
    unsafe { run_flat_alloc(env, z, kernel) }
}

/// Java: `PithMath.complexSqrt(double[] z) → double[]`.
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_complexSqrt(
    env: RawEnv,
    _class: JClass,
    z: JDoubleArray,
) -> JDoubleArray {
    let kernel = |x: &[f64]| -> Vec<f64> {
        if x.len() != 2 {
            return Vec::new();
        }
        let r = Complex::new(x[0], x[1]).sqrt();
        vec![r.re, r.im]
    };
    unsafe { run_flat_alloc(env, z, kernel) }
}

/// Java: `PithMath.complexLog(double[] z) → double[]` — the principal
/// `[ln|z|, arg z]`; `z = 0` surfaces as `NULL` (domain refusal).
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_complexLog(
    env: RawEnv,
    _class: JClass,
    z: JDoubleArray,
) -> JDoubleArray {
    let kernel = |x: &[f64]| -> Vec<f64> {
        if x.len() != 2 || (x[0] == 0.0 && x[1] == 0.0) {
            return Vec::new();
        }
        let w = Complex::new(x[0], x[1]).ln();
        vec![w.re, w.im]
    };
    unsafe { run_flat_alloc(env, z, kernel) }
}

/// Java: `PithMath.complexArg(double[] z) → double`.
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_complexArg(
    env: RawEnv,
    _class: JClass,
    z: JDoubleArray,
) -> f64 {
    let kernel =
        |x: &[f64]| -> Option<f64> { (x.len() == 2).then(|| Complex::new(x[0], x[1]).arg()) };
    unsafe { run_flat_scalar(env, z, kernel) }
}

/// Java: `PithMath.complexPowi(double[] z, int n) → double[]`.
///
/// # Safety
///
/// See [`Java_pith_math_PithMath_dct2`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_pith_math_PithMath_complexPowi(
    env: RawEnv,
    _class: JClass,
    z: JDoubleArray,
    n: i32,
) -> JDoubleArray {
    let kernel = move |x: &[f64]| -> Vec<f64> {
        if x.len() != 2 {
            return Vec::new();
        }
        let r = Complex::new(x[0], x[1]).powi(n);
        vec![r.re, r.im]
    };
    unsafe { run_flat_alloc(env, z, kernel) }
}

// -- shape helpers --------------------------------------------------------

/// Reassembles a row-major flat buffer into `[[f64; 3]; 3]`. Every
/// caller validates the length before calling.
fn matrix3(flat: &[f64]) -> [[f64; 3]; 3] {
    core::array::from_fn(|r| core::array::from_fn(|c| flat[r * 3 + c]))
}

/// Flattens a matrix back to row-major `f64`s.
fn flat3(m: &[[f64; 3]; 3]) -> Vec<f64> {
    m.iter().flat_map(|row| row.iter().copied()).collect()
}
