//! The fake-env JNI battery: every `Java_pith_math_PithMath_*` export
//! driven through a synthetic JNI function table — no JVM.
//!
//! The fake table installs stubs at the exact indices
//! [`pith_math::ffi_jni`] addresses, backed by a process-wide handle →
//! `Vec<f64>` map. This is both the correctness check for the JNI
//! glue and the execution pass the coverage gate needs: exports that
//! are never called do not count as covered lines.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use pith_math::ffi_jni::{
    Java_pith_math_PithMath_complexArg, Java_pith_math_PithMath_complexDiv,
    Java_pith_math_PithMath_complexExp, Java_pith_math_PithMath_complexLog,
    Java_pith_math_PithMath_complexMul, Java_pith_math_PithMath_complexPowi,
    Java_pith_math_PithMath_complexSqrt, Java_pith_math_PithMath_conv,
    Java_pith_math_PithMath_corr, Java_pith_math_PithMath_covariance, Java_pith_math_PithMath_dct2,
    Java_pith_math_PithMath_dct3, Java_pith_math_PithMath_dct22d, Java_pith_math_PithMath_dct32d,
    Java_pith_math_PithMath_det3, Java_pith_math_PithMath_dtw, Java_pith_math_PithMath_fft,
    Java_pith_math_PithMath_fftN, Java_pith_math_PithMath_fftReal, Java_pith_math_PithMath_idct2,
    Java_pith_math_PithMath_ifft, Java_pith_math_PithMath_ifftN, Java_pith_math_PithMath_inverse3,
    Java_pith_math_PithMath_lagrange, Java_pith_math_PithMath_mat3Mul,
    Java_pith_math_PithMath_mat3MulVec, Java_pith_math_PithMath_mean,
    Java_pith_math_PithMath_median, Java_pith_math_PithMath_ransacLine,
    Java_pith_math_PithMath_solve3, Java_pith_math_PithMath_transpose3,
    Java_pith_math_PithMath_variance,
};

type Env = *const *const [usize; 256];
type Arr = *mut core::ffi::c_void;

const IDX_GET_ARRAY_LENGTH: usize = 171;
const IDX_NEW_DOUBLE_ARRAY: usize = 182;
const IDX_GET_DOUBLE_ARRAY_ELEMENTS: usize = 190;
const IDX_RELEASE_DOUBLE_ARRAY_ELEMENTS: usize = 198;
const IDX_SET_DOUBLE_ARRAY_REGION: usize = 214;

/// Handle → contents for every array the fake JVM has allocated.
static ARRAYS: Mutex<Option<HashMap<usize, Vec<f64>>>> = Mutex::new(None);

fn with_maps<R>(f: impl FnOnce(&mut HashMap<usize, Vec<f64>>) -> R) -> R {
    let mut guard = ARRAYS.lock().unwrap();
    f(guard.get_or_insert_with(HashMap::new))
}

unsafe extern "system" fn stub_len(_env: Env, arr: Arr, _copy: *mut u8) -> i32 {
    with_maps(|m| m.get(&(arr as usize)).map_or(0, |v| v.len() as i32))
}

unsafe extern "system" fn stub_new(_env: Env, len: i32) -> Arr {
    with_maps(|m| {
        let handle = (0x5000_0000usize + m.len() * 8 + 8) | 1;
        m.insert(handle, vec![0.0; len.max(0) as usize]);
        handle as Arr
    })
}

unsafe extern "system" fn stub_elements(_env: Env, arr: Arr, _copy: *mut u8) -> *mut f64 {
    with_maps(|m| {
        m.get(&(arr as usize))
            .map_or(core::ptr::null_mut(), |v| v.as_ptr() as *mut f64)
    })
}

unsafe extern "system" fn stub_release(_env: Env, _arr: Arr, _ptr: *mut f64, _mode: i32) {}

unsafe extern "system" fn stub_set_region(
    _env: Env,
    arr: Arr,
    start: i32,
    len: i32,
    data: *const f64,
) {
    let src = unsafe { core::slice::from_raw_parts(data, len.max(0) as usize) };
    with_maps(|m| {
        if let Some(v) = m.get_mut(&(arr as usize)) {
            let s = start.max(0) as usize;
            v[s..s + src.len()].copy_from_slice(src);
        }
    })
}

/// A raw table pointer wrapper: immutable after init, so sharing
/// across test threads is sound.
struct TablePtr(*const [usize; 256]);
unsafe impl Send for TablePtr {}
unsafe impl Sync for TablePtr {}

/// The fake env, owned for the process lifetime by the static: the
/// tuple keeps the table alive and `FAKE.1` carries the pointer the
/// JNI glue dereferences once to reach the table.
static FAKE: LazyLock<(Box<[usize; 256]>, TablePtr)> = LazyLock::new(|| {
    let mut table = Box::new([0usize; 256]);
    table[IDX_GET_ARRAY_LENGTH] = stub_len as *const () as usize;
    table[IDX_NEW_DOUBLE_ARRAY] = stub_new as *const () as usize;
    table[IDX_GET_DOUBLE_ARRAY_ELEMENTS] = stub_elements as *const () as usize;
    table[IDX_RELEASE_DOUBLE_ARRAY_ELEMENTS] = stub_release as *const () as usize;
    table[IDX_SET_DOUBLE_ARRAY_REGION] = stub_set_region as *const () as usize;
    let ptr: *const [usize; 256] = &*table;
    (table, TablePtr(ptr))
});

/// The fake env handle handed to every export under test.
fn env() -> Env {
    &FAKE.1.0 as *const *const [usize; 256]
}

/// Registers `vals` and returns its fake array handle.
fn arr(vals: &[f64]) -> Arr {
    with_maps(|m| {
        let handle = (0x5000_0000usize + m.len() * 8 + 8) | 1;
        m.insert(handle, vals.to_vec());
        handle as Arr
    })
}

/// Reads back a fake array's contents.
fn read(arr: Arr) -> Vec<f64> {
    with_maps(|m| m.get(&(arr as usize)).cloned().unwrap_or_default())
}

const NULL: Arr = core::ptr::null_mut();

#[test]
fn dct_and_fft_surfaces_round_trip() {
    let e = env();
    let x = arr(&[0.5, -1.25, 2.0, 3.75, -0.125, 1.5, -2.5, 0.75]);
    let c8: Vec<f64> = (0..16).map(|i| i as f64 * 0.25 - 1.5).collect();
    let xc = arr(&c8);

    let out = unsafe { Java_pith_math_PithMath_dct2(e, NULL, x) };
    assert_eq!(read(out).len(), 8);
    let out = unsafe { Java_pith_math_PithMath_idct2(e, NULL, x) };
    assert_eq!(read(out).len(), 8);
    let out = unsafe { Java_pith_math_PithMath_dct3(e, NULL, x) };
    assert_eq!(read(out).len(), 8);
    // idct2(dct2(x)) == x, through the JNI surface.
    let fwd = unsafe { Java_pith_math_PithMath_dct2(e, NULL, x) };
    let back = unsafe { Java_pith_math_PithMath_idct2(e, NULL, fwd) };
    for (g, w) in read(back)
        .iter()
        .zip([0.5, -1.25, 2.0, 3.75, -0.125, 1.5, -2.5, 0.75])
    {
        assert!((g - w).abs() < 1e-13);
    }

    let m64: Vec<f64> = (0..64).map(|i| ((i % 7) as f64) - 2.0).collect();
    let xm = arr(&m64);
    let out = unsafe { Java_pith_math_PithMath_dct22d(e, NULL, xm, 8, 8) };
    assert_eq!(read(out).len(), 64);

    let out = unsafe { Java_pith_math_PithMath_fft(e, NULL, xc) };
    assert_eq!(read(out).len(), 16);
    let out = unsafe { Java_pith_math_PithMath_ifft(e, NULL, xc) };
    assert_eq!(read(out).len(), 16);
    let out = unsafe { Java_pith_math_PithMath_fftReal(e, NULL, x) };
    assert_eq!(read(out).len(), 16);

    // fftN on n = 12 — a length the radix-2 surface refuses.
    let c12: Vec<f64> = (0..24).map(|i| i as f64 * 0.125 - 1.0).collect();
    let x12 = arr(&c12);
    let out = unsafe { Java_pith_math_PithMath_fftN(e, NULL, x12) };
    assert_eq!(read(out).len(), 24);
}

#[test]
fn linalg_and_median_surfaces_match_the_reference_values() {
    let e = env();
    let det_in = arr(&[6.0, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0]);
    assert_eq!(
        unsafe { Java_pith_math_PithMath_det3(e, NULL, det_in) },
        -306.0
    );

    let m = arr(&[2.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 2.0]);
    let inv = unsafe { Java_pith_math_PithMath_inverse3(e, NULL, m) };
    assert_eq!(read(inv).len(), 9);
    let tr = unsafe { Java_pith_math_PithMath_transpose3(e, NULL, m) };
    assert_eq!(read(tr), vec![2.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 2.0]);

    // solve3 on the textbook system: answer 2, 3, −1 within ulps.
    let a = arr(&[2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0]);
    let b = arr(&[8.0, -11.0, -3.0]);
    let x = unsafe { Java_pith_math_PithMath_solve3(e, NULL, a, b) };
    let got = read(x);
    assert_eq!(got.len(), 3);
    for (g, w) in got.iter().zip([2.0, 3.0, -1.0]) {
        assert!((g - w).abs() < 1e-12);
    }

    let prod = unsafe { Java_pith_math_PithMath_mat3Mul(e, NULL, m, m) };
    assert_eq!(read(prod).len(), 9);
    let v = arr(&[1.0, 2.0, 3.0]);
    let mv = unsafe { Java_pith_math_PithMath_mat3MulVec(e, NULL, m, v) };
    assert_eq!(read(mv).len(), 3);

    let med = unsafe { Java_pith_math_PithMath_median(e, NULL, arr(&[1.0, 2.0, 3.0, 4.0])) };
    assert_eq!(med, 2.0);
}

#[test]
fn tier1_surfaces_produce_the_exact_vectors() {
    let e = env();

    // conv / corr.
    let a = arr(&[1.0, 2.0, 3.0]);
    let b = arr(&[0.0, 1.0, 2.0]);
    let got = read(unsafe { Java_pith_math_PithMath_conv(e, NULL, a, b) });
    assert_eq!(got, vec![0.0, 1.0, 4.0, 7.0, 6.0]);
    let got =
        read(unsafe { Java_pith_math_PithMath_corr(e, NULL, arr(&[1.0, 2.0]), arr(&[3.0, 4.0])) });
    assert_eq!(got, vec![4.0, 11.0, 6.0]);

    // stats.
    assert_eq!(
        unsafe { Java_pith_math_PithMath_mean(e, NULL, arr(&[1.0, 2.0, 3.0, 4.0])) },
        2.5
    );
    let v = unsafe { Java_pith_math_PithMath_variance(e, NULL, arr(&[1.0, 3.0])) };
    assert_eq!(v.to_bits(), 2.0_f64.to_bits());
    let c = unsafe {
        Java_pith_math_PithMath_covariance(e, NULL, arr(&[1.0, 2.0, 3.0]), arr(&[2.0, 4.0, 6.0]))
    };
    assert_eq!(c.to_bits(), 2.0_f64.to_bits());

    // lagrange through (0,1),(1,4),(2,9) at 1.5 = 6.25 exactly.
    let l = unsafe {
        Java_pith_math_PithMath_lagrange(e, NULL, arr(&[0.0, 1.0, 2.0]), arr(&[1.0, 4.0, 9.0]), 1.5)
    };
    assert_eq!(l.to_bits(), 6.25_f64.to_bits());

    // dtw textbook.
    let d = unsafe {
        Java_pith_math_PithMath_dtw(e, NULL, arr(&[1.0, 2.0, 3.0]), arr(&[2.0, 2.0, 2.0]))
    };
    assert_eq!(d, 2.0);

    // ransac: the planted line, exact model out.
    let fit = read(unsafe {
        Java_pith_math_PithMath_ransacLine(
            e,
            NULL,
            arr(&[0.0, 2.0, 4.0, 6.0, 2.0, 4.0]),
            arr(&[1.0, 2.0, 3.0, 4.0, 10.0, -9.0]),
            0.5,
            64,
            42,
        )
    });
    assert_eq!(fit, vec![0.5, 1.0, 4.0]);

    // complex ops.
    let got = read(unsafe {
        Java_pith_math_PithMath_complexMul(e, NULL, arr(&[3.0, 2.0]), arr(&[1.0, -4.0]))
    });
    assert_eq!(got, vec![11.0, -10.0]);
    let got = read(unsafe {
        Java_pith_math_PithMath_complexDiv(e, NULL, arr(&[5.0, 1.0]), arr(&[1.0, -1.0]))
    });
    assert_eq!(got, vec![2.0, 3.0]);
    let got = read(unsafe { Java_pith_math_PithMath_complexExp(e, NULL, arr(&[0.0, 0.0])) });
    assert_eq!(got, vec![1.0, 0.0]);
    let got = read(unsafe { Java_pith_math_PithMath_complexSqrt(e, NULL, arr(&[4.0, 0.0])) });
    assert_eq!(got, vec![2.0, 0.0]);
    let got = read(unsafe { Java_pith_math_PithMath_complexPowi(e, NULL, arr(&[1.0, 1.0]), 4) });
    assert_eq!(got, vec![-4.0, 0.0]);
    let got = unsafe { Java_pith_math_PithMath_complexArg(e, NULL, arr(&[1.0, 1.0])) };
    assert_eq!(got, core::f64::consts::FRAC_PI_4);
}

#[test]
fn refusal_conventions_hold_across_the_surface() {
    let e = env();

    // NULL arrays → NULL / NaN, never a crash.
    assert!(unsafe { Java_pith_math_PithMath_dct2(e, NULL, NULL) }.is_null());
    assert!(unsafe { Java_pith_math_PithMath_solve3(e, NULL, NULL, NULL) }.is_null());
    assert!(unsafe { Java_pith_math_PithMath_conv(e, NULL, NULL, NULL) }.is_null());
    assert!(
        unsafe { Java_pith_math_PithMath_ransacLine(e, NULL, NULL, NULL, 1.0, 8, 1) }.is_null()
    );
    assert!(unsafe { Java_pith_math_PithMath_mean(e, NULL, NULL) }.is_nan());
    assert!(unsafe { Java_pith_math_PithMath_det3(e, NULL, NULL) }.is_nan());
    assert!(unsafe { Java_pith_math_PithMath_dtw(e, NULL, NULL, NULL) }.is_nan());
    assert!(unsafe { Java_pith_math_PithMath_lagrange(e, NULL, NULL, NULL, 0.5) }.is_nan());
    assert!(unsafe { Java_pith_math_PithMath_complexArg(e, NULL, NULL) }.is_nan());

    // Empty inputs are the Java pre-validation's domain, mirrored: NULL.
    let empty = arr(&[]);
    assert!(unsafe { Java_pith_math_PithMath_dct3(e, NULL, empty) }.is_null());
    assert!(unsafe { Java_pith_math_PithMath_conv(e, NULL, empty, arr(&[1.0])) }.is_null());
    assert!(unsafe { Java_pith_math_PithMath_fftReal(e, NULL, empty) }.is_null());
    assert!(unsafe { Java_pith_math_PithMath_fftN(e, NULL, empty) }.is_null());

    // Broken geometry: fft on a non-power-of-two, dct2_2d mismatched,
    // linalg with wrong lengths.
    let c6 = arr(&[0.0; 12]); // 6 complex pairs — not a power of two.
    assert!(unsafe { Java_pith_math_PithMath_fft(e, NULL, c6) }.is_null());
    assert!(unsafe { Java_pith_math_PithMath_ifft(e, NULL, c6) }.is_null());
    let odd_complex = arr(&[0.0; 6]); // 3 pairs — odd.
    assert!(unsafe { Java_pith_math_PithMath_fft(e, NULL, odd_complex) }.is_null());
    let m64 = arr(&[0.0; 64]);
    assert!(unsafe { Java_pith_math_PithMath_dct22d(e, NULL, m64, 4, 8) }.is_null());
    let short = arr(&[1.0, 2.0]);
    assert!(unsafe { Java_pith_math_PithMath_inverse3(e, NULL, short) }.is_null());
    assert!(unsafe { Java_pith_math_PithMath_mat3Mul(e, NULL, short, short) }.is_null());
    assert!(unsafe { Java_pith_math_PithMath_mat3MulVec(e, NULL, m64, arr(&[1.0])) }.is_null());
    assert!(unsafe { Java_pith_math_PithMath_det3(e, NULL, short) }.is_nan());
    assert!(unsafe { Java_pith_math_PithMath_solve3(e, NULL, short, arr(&[1.0])) }.is_null());

    // Data refusals: singular solve3 → NULL; duplicated Lagrange nodes
    // → NaN; all-vertical RANSAC → NULL; covariance length mismatch
    // → NaN.
    let zeros = arr(&[0.0; 9]);
    assert!(unsafe { Java_pith_math_PithMath_solve3(e, NULL, zeros, arr(&[0.0; 3])) }.is_null());
    assert!(unsafe { Java_pith_math_PithMath_inverse3(e, NULL, zeros) }.is_null());
    let l = unsafe {
        Java_pith_math_PithMath_lagrange(e, NULL, arr(&[1.0, 1.0]), arr(&[2.0, 3.0]), 0.5)
    };
    assert!(l.is_nan());
    let fit = unsafe {
        Java_pith_math_PithMath_ransacLine(
            e,
            NULL,
            arr(&[2.0, 2.0, 2.0]),
            arr(&[1.0, 5.0, 9.0]),
            0.5,
            64,
            1,
        )
    };
    assert!(fit.is_null());
    assert!(
        unsafe {
            Java_pith_math_PithMath_ransacLine(e, NULL, arr(&[0.0]), arr(&[1.0]), 0.5, 64, 1)
        }
        .is_null()
    );
    assert!(
        unsafe {
            Java_pith_math_PithMath_ransacLine(
                e,
                NULL,
                arr(&[0.0, 1.0]),
                arr(&[1.0, 2.0]),
                0.0,
                64,
                1,
            )
        }
        .is_null()
    );
    assert!(
        unsafe {
            Java_pith_math_PithMath_ransacLine(
                e,
                NULL,
                arr(&[0.0, 1.0]),
                arr(&[1.0, 2.0]),
                0.5,
                0,
                1,
            )
        }
        .is_null()
    );
    assert!(
        unsafe { Java_pith_math_PithMath_covariance(e, NULL, arr(&[1.0]), arr(&[1.0, 2.0])) }
            .is_nan()
    );
    assert!(unsafe { Java_pith_math_PithMath_variance(e, NULL, arr(&[1.0])) }.is_nan());
    assert!(unsafe { Java_pith_math_PithMath_mean(e, NULL, empty) }.is_nan());
    assert!(unsafe { Java_pith_math_PithMath_dtw(e, NULL, empty, arr(&[1.0])) }.is_nan());
    // Wrong complex pair counts → NULL/NaN.
    assert!(
        unsafe { Java_pith_math_PithMath_complexMul(e, NULL, arr(&[1.0]), arr(&[1.0, 2.0])) }
            .is_null()
    );
    assert!(
        unsafe { Java_pith_math_PithMath_complexExp(e, NULL, arr(&[1.0, 2.0, 3.0])) }.is_null()
    );
    assert!(
        unsafe { Java_pith_math_PithMath_complexSqrt(e, NULL, arr(&[1.0, 2.0, 3.0])) }.is_null()
    );
    assert!(
        unsafe { Java_pith_math_PithMath_complexPowi(e, NULL, arr(&[1.0, 2.0, 3.0]), 2) }.is_null()
    );
    assert!(unsafe { Java_pith_math_PithMath_complexArg(e, NULL, arr(&[1.0, 2.0, 3.0])) }.is_nan());
    // Median of nothing.
    assert!(unsafe { Java_pith_math_PithMath_median(e, NULL, empty) }.is_nan());

    // Remaining geometry refusals across the surface.
    assert!(unsafe { Java_pith_math_PithMath_dct22d(e, NULL, m64, 0, 8) }.is_null());
    assert!(unsafe { Java_pith_math_PithMath_dct22d(e, NULL, m64, 8, 0) }.is_null());
    let r6 = arr(&[0.0; 6]); // non-power-of-two real length
    assert!(unsafe { Java_pith_math_PithMath_fftReal(e, NULL, r6) }.is_null());
    assert!(unsafe { Java_pith_math_PithMath_fftN(e, NULL, arr(&[0.0; 5])) }.is_null());
    assert!(unsafe { Java_pith_math_PithMath_transpose3(e, NULL, short) }.is_null());
    let lag =
        unsafe { Java_pith_math_PithMath_lagrange(e, NULL, arr(&[0.0, 1.0]), arr(&[1.0]), 0.5) };
    assert!(lag.is_nan());
    assert!(
        unsafe {
            Java_pith_math_PithMath_complexDiv(e, NULL, arr(&[1.0, 2.0]), arr(&[1.0, 2.0, 3.0]))
        }
        .is_null()
    );
    assert!(unsafe { Java_pith_math_PithMath_ifftN(e, NULL, arr(&[0.0; 5])) }.is_null());
    assert!(unsafe { Java_pith_math_PithMath_dct32d(e, NULL, m64, 0, 8) }.is_null());
    assert!(unsafe { Java_pith_math_PithMath_dct32d(e, NULL, m64, 4, 8) }.is_null());
    assert!(unsafe { Java_pith_math_PithMath_complexLog(e, NULL, arr(&[0.0, 0.0])) }.is_null());

    // Happy paths for the newest exports.
    let x12 = arr(&[
        1.0, 0.0, 0.5, -0.5, -1.0, 2.0, 0.25, 1.5, -0.75, 0.0, 2.0, -1.0,
    ]);
    let fwd = unsafe { Java_pith_math_PithMath_fftN(e, NULL, x12) };
    let inv = unsafe { Java_pith_math_PithMath_ifftN(e, NULL, fwd) };
    let rt = read(inv);
    assert_eq!(rt.len(), 12);
    assert!((rt[0] - 1.0).abs() < 1e-12 && rt[1].abs() < 1e-12);
    let m64f: Vec<f64> = (0..64).map(|i| ((i % 7) as f64) - 2.0).collect();
    let m16 = unsafe { Java_pith_math_PithMath_dct32d(e, NULL, arr(&m64f), 8, 8) };
    assert_eq!(read(m16).len(), 64);
    let lg =
        unsafe { Java_pith_math_PithMath_complexLog(e, NULL, arr(&[std::f64::consts::E, 0.0])) };
    let lv = read(lg);
    assert_eq!(lv.len(), 2);
    assert!((lv[0] - 1.0).abs() < 1e-15 && lv[1].abs() < 1e-15);
}
