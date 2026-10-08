//! Integration-level exercise of every `pith_math::ffi` export.
//!
//! The coverage gate measures this target too: the `#[unsafe(no_mangle)]`
//! cdylib surface must be driven from outside the lib's own unit-test
//! binary, or the gate's merged report shows the exports as
//! linked-but-unexecuted (the unit tests alone do not carry the
//! `--all-targets` profile for them). Everything here mirrors the
//! in-crate `ffi::tests` from the outside: happy paths through raw
//! pointers incl. the free round-trip, plus every refusal class.

use pith_math::ffi::{
    PITH_E_INVALID, PITH_E_REJECTED, PITH_OK, pith_math_complex_arg, pith_math_complex_div,
    pith_math_complex_exp, pith_math_complex_mul, pith_math_complex_powi, pith_math_complex_sqrt,
    pith_math_conv, pith_math_corr, pith_math_cov, pith_math_dct2, pith_math_dct2_2d,
    pith_math_dct3, pith_math_det3, pith_math_dtw, pith_math_fft, pith_math_fft_n,
    pith_math_fft_real, pith_math_free, pith_math_idct2, pith_math_ifft, pith_math_inverse3,
    pith_math_lagrange, pith_math_mat3_mul, pith_math_mat3_mul_vec, pith_math_mean,
    pith_math_median, pith_math_ransac_line, pith_math_solve3, pith_math_transpose3, pith_math_var,
};

/// Calls an allocating export with a valid input, copies the handed-out
/// buffer out (element count = bytes / 8) and frees it.
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

/// Calls a scalar-out export.
fn call_scalar(op: impl Fn(*const f64, usize, *mut f64) -> i32, input: &[f64]) -> (i32, f64) {
    let mut slot: f64 = 0.0;
    (op(input.as_ptr(), input.len(), &mut slot), slot)
}

/// Calls an allocating export with one extra `usize` (a split/geometry
/// argument) and a valid input, copying the handed-out buffer out.
fn call_alloc_split(
    op: impl Fn(*const f64, usize, usize, *mut *mut f64, *mut usize) -> i32,
    input: &[f64],
    extra: usize,
) -> (i32, Vec<f64>) {
    let mut out: *mut f64 = core::ptr::null_mut();
    let mut out_len: usize = 0;
    let status = op(input.as_ptr(), input.len(), extra, &mut out, &mut out_len);
    if status != PITH_OK {
        return (status, Vec::new());
    }
    let count = out_len / core::mem::size_of::<f64>();
    let copied = unsafe { core::slice::from_raw_parts(out, count) }.to_vec();
    unsafe { pith_math_free(out, out_len) };
    (status, copied)
}

/// Calls an allocating export with the RANSAC argument tail.
#[allow(clippy::too_many_arguments)]
fn call_alloc_ransac(
    op: impl Fn(*const f64, usize, f64, usize, u64, *mut *mut f64, *mut usize) -> i32,
    input: &[f64],
    threshold: f64,
    iterations: usize,
    seed: u64,
) -> (i32, Vec<f64>) {
    let mut out: *mut f64 = core::ptr::null_mut();
    let mut out_len: usize = 0;
    let status = op(
        input.as_ptr(),
        input.len(),
        threshold,
        iterations,
        seed,
        &mut out,
        &mut out_len,
    );
    if status != PITH_OK {
        return (status, Vec::new());
    }
    let count = out_len / core::mem::size_of::<f64>();
    let copied = unsafe { core::slice::from_raw_parts(out, count) }.to_vec();
    unsafe { pith_math_free(out, out_len) };
    (status, copied)
}

#[test]
fn every_alloc_export_runs_from_an_integration_binary() {
    let x8 = [0.5, -1.25, 2.0, 3.75, -0.125, 1.5, -2.5, 0.75];
    let c8: Vec<f64> = (0..16).map(|i| (i as f64) * 0.25 - 1.5).collect();
    let m64: Vec<f64> = (0..64).map(|i| ((i % 7) as f64) - 2.0).collect();
    let m = [[2.0, 0.0, 1.0], [0.0, 3.0, 0.0], [1.0, 0.0, 2.0]];
    let flat_m: Vec<f64> = m.iter().flat_map(|r| r.iter().copied()).collect();

    let (status, got) = call_alloc(|i, n, o, l| unsafe { pith_math_dct2(i, n, o, l) }, &x8);
    assert_eq!((status, got.len()), (PITH_OK, 8));

    let (status, got) = call_alloc(|i, n, o, l| unsafe { pith_math_idct2(i, n, o, l) }, &x8);
    assert_eq!((status, got.len()), (PITH_OK, 8));

    let (status, got) = call_alloc(
        |i, n, o, l| unsafe { pith_math_dct2_2d(i, n, 8, 8, o, l) },
        &m64,
    );
    assert_eq!((status, got.len()), (PITH_OK, 64));

    let (status, got) = call_alloc(|i, n, o, l| unsafe { pith_math_fft(i, n, o, l) }, &c8);
    assert_eq!((status, got.len()), (PITH_OK, 16));

    let (status, got) = call_alloc(|i, n, o, l| unsafe { pith_math_ifft(i, n, o, l) }, &c8);
    assert_eq!((status, got.len()), (PITH_OK, 16));

    let (status, got) = call_alloc(|i, n, o, l| unsafe { pith_math_fft_real(i, n, o, l) }, &x8);
    assert_eq!((status, got.len()), (PITH_OK, 16));

    let system: Vec<f64> = flat_m.iter().copied().chain([8.0, -11.0, -3.0]).collect();
    let (status, got) = call_alloc(
        |i, n, o, l| unsafe { pith_math_solve3(i, n, o, l) },
        &system,
    );
    assert_eq!((status, got.len()), (PITH_OK, 3));

    let (status, got) = call_alloc(
        |i, n, o, l| unsafe { pith_math_inverse3(i, n, o, l) },
        &flat_m,
    );
    assert_eq!((status, got.len()), (PITH_OK, 9));

    let (status, got) = call_alloc(
        |i, n, o, l| unsafe { pith_math_transpose3(i, n, o, l) },
        &flat_m,
    );
    assert_eq!((status, got.len()), (PITH_OK, 9));

    let (status, got) = call_alloc(
        |i, n, o, l| unsafe { pith_math_mat3_mul(i, n, o, l) },
        &flat_m
            .iter()
            .copied()
            .chain(flat_m.iter().copied())
            .collect::<Vec<f64>>(),
    );
    assert_eq!((status, got.len()), (PITH_OK, 9));

    let (status, got) = call_alloc(
        |i, n, o, l| unsafe { pith_math_mat3_mul_vec(i, n, o, l) },
        &flat_m
            .iter()
            .copied()
            .chain([1.0, 2.0, 3.0])
            .collect::<Vec<f64>>(),
    );
    assert_eq!((status, got.len()), (PITH_OK, 3));
}

#[test]
fn every_scalar_export_and_refusal_runs_from_an_integration_binary() {
    // det3.standard, pinned from tests/reference.json.
    let det_in: Vec<f64> = [
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
    let (status, det) = call_scalar(|i, n, o| unsafe { pith_math_det3(i, n, o) }, &det_in);
    assert_eq!((status, det.to_bits()), (PITH_OK, 0xc073200000000000));

    let (status, med) = call_scalar(
        |i, n, o| unsafe { pith_math_median(i, n, o) },
        &[1.0, 2.0, 3.0, 4.0],
    );
    assert_eq!((status, med), (PITH_OK, 2.0));

    // Refusals: null data, broken geometry, empty inputs, singular
    // systems — status codes, never panics.
    let mut out: *mut f64 = core::ptr::null_mut();
    let mut out_len: usize = 0;
    let mut slot: f64 = 0.0;
    assert_eq!(
        unsafe { pith_math_dct2(core::ptr::null(), 0, &mut out, &mut out_len) },
        PITH_E_INVALID
    );
    assert_eq!(
        unsafe { pith_math_det3(core::ptr::null(), 0, &mut slot) },
        PITH_E_INVALID
    );
    assert_eq!(
        call_alloc(|i, n, o, l| unsafe { pith_math_fft(i, n, o, l) }, &[0.0; 6]).0,
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
    assert_eq!(
        call_alloc(
            |i, n, o, l| unsafe { pith_math_fft_real(i, n, o, l) },
            &[0.0; 3]
        )
        .0,
        PITH_E_INVALID
    );
    assert_eq!(
        call_alloc(|i, n, o, l| unsafe { pith_math_dct2(i, n, o, l) }, &[]).0,
        PITH_E_INVALID
    );
    assert_eq!(
        call_alloc(|i, n, o, l| unsafe { pith_math_idct2(i, n, o, l) }, &[]).0,
        PITH_E_INVALID
    );
    assert_eq!(
        call_alloc(
            |i, n, o, l| unsafe { pith_math_dct2_2d(i, n, 4, 8, o, l) },
            &[0.0; 64]
        )
        .0,
        PITH_E_INVALID
    );
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
    assert_eq!(
        call_scalar(|i, n, o| unsafe { pith_math_median(i, n, o) }, &[]).0,
        PITH_E_INVALID
    );
    assert_eq!(
        call_scalar(|i, n, o| unsafe { pith_math_det3(i, n, o) }, &[0.0; 8]).0,
        PITH_E_INVALID
    );

    // A null buffer is a legal free.
    unsafe { pith_math_free(core::ptr::null_mut(), 0) };
}

// ---------------------------------------------------------------------
// Tier-1 exports
// ---------------------------------------------------------------------

#[test]
fn tier1_alloc_exports_run_from_an_integration_binary() {
    // Bluestein on n = 12 (a length radix-2 refuses).
    let c12: Vec<f64> = (0..24).map(|i| (i as f64) * 0.25 - 2.5).collect();
    let (status, got) = call_alloc(|i, n, o, l| unsafe { pith_math_fft_n(i, n, o, l) }, &c12);
    assert_eq!((status, got.len()), (PITH_OK, 24));

    let x8 = [0.5, -1.25, 2.0, 3.75, -0.125, 1.5, -2.5, 0.75];
    let (status, got) = call_alloc(|i, n, o, l| unsafe { pith_math_dct3(i, n, o, l) }, &x8);
    assert_eq!((status, got.len()), (PITH_OK, 8));

    // conv: [1,2,3] ∗ [0,1,2] — packed with a_len = 3.
    let (status, got) = call_alloc_split(
        |i, n, a, o, l| unsafe { pith_math_conv(i, n, a, o, l) },
        &[1.0, 2.0, 3.0, 0.0, 1.0, 2.0],
        3,
    );
    assert_eq!((status, got), (PITH_OK, vec![0.0, 1.0, 4.0, 7.0, 6.0]));

    // corr: [1,2] ⋆ [3,4] — packed with a_len = 2.
    let (status, got) = call_alloc_split(
        |i, n, a, o, l| unsafe { pith_math_corr(i, n, a, o, l) },
        &[1.0, 2.0, 3.0, 4.0],
        2,
    );
    assert_eq!((status, got), (PITH_OK, vec![4.0, 11.0, 6.0]));

    // ransac_line: the planted y = 0.5x + 1 vector, exact model out.
    let pts: Vec<f64> = [0.0, 1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 2.0, 10.0, 4.0, -9.0].to_vec();
    let (status, got) = call_alloc_ransac(
        |i, n, t, it, sd, o, l| unsafe { pith_math_ransac_line(i, n, t, it, sd, o, l) },
        &pts,
        0.5,
        64,
        42,
    );
    assert_eq!((status, got), (PITH_OK, vec![0.5, 1.0, 4.0]));

    // complex mul / div: exact integer identities.
    let (status, got) = call_alloc(
        |i, n, o, l| unsafe { pith_math_complex_mul(i, n, o, l) },
        &[3.0, 2.0, 1.0, -4.0],
    );
    assert_eq!((status, got), (PITH_OK, vec![11.0, -10.0]));
    let (status, got) = call_alloc(
        |i, n, o, l| unsafe { pith_math_complex_div(i, n, o, l) },
        &[5.0, 1.0, 1.0, -1.0],
    );
    assert_eq!((status, got), (PITH_OK, vec![2.0, 3.0]));

    let (status, got) = call_alloc(
        |i, n, o, l| unsafe { pith_math_complex_exp(i, n, o, l) },
        &[0.0, 0.0],
    );
    assert_eq!((status, got), (PITH_OK, vec![1.0, 0.0]));

    let (status, got) = call_alloc(
        |i, n, o, l| unsafe { pith_math_complex_sqrt(i, n, o, l) },
        &[4.0, 0.0],
    );
    assert_eq!((status, got), (PITH_OK, vec![2.0, 0.0]));

    let (status, got) = call_alloc_split(
        |i, n, p, o, l| unsafe { pith_math_complex_powi(i, n, p, o, l) },
        &[1.0, 1.0],
        4,
    );
    assert_eq!((status, got), (PITH_OK, vec![-4.0, 0.0]));
}

#[test]
fn tier1_scalar_exports_run_from_an_integration_binary() {
    let (status, m) = call_scalar(
        |i, n, o| unsafe { pith_math_mean(i, n, o) },
        &[1.0, 2.0, 3.0, 4.0],
    );
    assert_eq!((status, m), (PITH_OK, 2.5));

    let (status, v) = call_scalar(|i, n, o| unsafe { pith_math_var(i, n, o) }, &[1.0, 3.0]);
    assert_eq!((status, v.to_bits()), (PITH_OK, 2.0_f64.to_bits()));

    // cov: (1,2,3) vs (2,4,6) packed — exactly 2.
    let (status, c) = call_scalar(
        |i, n, o| unsafe { pith_math_cov(i, n, o) },
        &[1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
    );
    assert_eq!((status, c.to_bits()), (PITH_OK, 2.0_f64.to_bits()));

    // lagrange through (0,1),(1,4),(2,9) at 1.5 = 6.25 exactly.
    let mut slot: f64 = 0.0;
    let pts = [0.0, 1.0, 1.0, 4.0, 2.0, 9.0];
    let status = unsafe { pith_math_lagrange(pts.as_ptr(), pts.len(), 1.5, &mut slot) };
    assert_eq!((status, slot.to_bits()), (PITH_OK, 6.25_f64.to_bits()));

    // dtw textbook: (1,2,3) vs (2,2,2) = 2, packed with a_len = 3.
    let (status, d) = call_scalar_with_split(
        |i, n, a, o| unsafe { pith_math_dtw(i, n, a, o) },
        &[1.0, 2.0, 3.0, 2.0, 2.0, 2.0],
        3,
    );
    assert_eq!((status, d.to_bits()), (PITH_OK, 2.0_f64.to_bits()));

    // complex arg(1+i) = π/4, the closest f64.
    let (status, a) = call_scalar(
        |i, n, o| unsafe { pith_math_complex_arg(i, n, o) },
        &[1.0, 1.0],
    );
    assert_eq!(a, core::f64::consts::FRAC_PI_4);
    let _ = status;
}

/// Calls a scalar-out export carrying a `usize` split argument.
fn call_scalar_with_split(
    op: impl Fn(*const f64, usize, usize, *mut f64) -> i32,
    input: &[f64],
    a_len: usize,
) -> (i32, f64) {
    let mut slot: f64 = 0.0;
    (op(input.as_ptr(), input.len(), a_len, &mut slot), slot)
}

#[test]
fn tier1_refusals_run_from_an_integration_binary() {
    // fft_n: odd interleaved count and n = 0 are invalid (n = 3 above
    // ran fine — any n ≥ 1 is legal here).
    assert_eq!(
        call_alloc(
            |i, n, o, l| unsafe { pith_math_fft_n(i, n, o, l) },
            &[0.0; 3]
        )
        .0,
        PITH_E_INVALID
    );
    assert_eq!(
        call_alloc(|i, n, o, l| unsafe { pith_math_fft_n(i, n, o, l) }, &[]).0,
        PITH_E_INVALID
    );
    assert_eq!(
        call_alloc(|i, n, o, l| unsafe { pith_math_dct3(i, n, o, l) }, &[]).0,
        PITH_E_INVALID
    );
    // conv: empty operand, a_len = 0, and a_len covering everything.
    assert_eq!(
        call_alloc_split(
            |i, n, a, o, l| unsafe { pith_math_conv(i, n, a, o, l) },
            &[0.0; 6],
            0
        )
        .0,
        PITH_E_INVALID
    );
    assert_eq!(
        call_alloc_split(
            |i, n, a, o, l| unsafe { pith_math_conv(i, n, a, o, l) },
            &[0.0; 6],
            6
        )
        .0,
        PITH_E_INVALID
    );
    // var: a single observation has no denominator.
    assert_eq!(
        call_scalar(|i, n, o| unsafe { pith_math_var(i, n, o) }, &[1.0]).0,
        PITH_E_INVALID
    );
    // cov: odd cut, and halves too short.
    assert_eq!(
        call_scalar(|i, n, o| unsafe { pith_math_cov(i, n, o) }, &[0.0; 3]).0,
        PITH_E_INVALID
    );
    assert_eq!(
        call_scalar(|i, n, o| unsafe { pith_math_cov(i, n, o) }, &[0.0; 2]).0,
        PITH_E_INVALID
    );
    // lagrange: duplicated abscissa is data-degenerate → REJECTED.
    let mut slot: f64 = 0.0;
    let dupes = [1.0, 2.0, 1.0, 3.0];
    assert_eq!(
        unsafe { pith_math_lagrange(dupes.as_ptr(), dupes.len(), 0.5, &mut slot) },
        PITH_E_REJECTED
    );
    // dtw: empty operand.
    assert_eq!(
        call_scalar_with_split(
            |i, n, a, o| unsafe { pith_math_dtw(i, n, a, o) },
            &[0.0; 3],
            3
        )
        .0,
        PITH_E_INVALID
    );
    // ransac: degenerate threshold, no iterations, too few points —
    // invalid; all-vertical points — rejected (no model found).
    assert_eq!(
        call_alloc_ransac(
            |i, n, t, it, sd, o, l| unsafe { pith_math_ransac_line(i, n, t, it, sd, o, l) },
            &[0.0; 4],
            0.0,
            64,
            1
        )
        .0,
        PITH_E_INVALID
    );
    assert_eq!(
        call_alloc_ransac(
            |i, n, t, it, sd, o, l| unsafe { pith_math_ransac_line(i, n, t, it, sd, o, l) },
            &[0.0; 4],
            0.5,
            0,
            1
        )
        .0,
        PITH_E_INVALID
    );
    assert_eq!(
        call_alloc_ransac(
            |i, n, t, it, sd, o, l| unsafe { pith_math_ransac_line(i, n, t, it, sd, o, l) },
            &[0.0; 2],
            0.5,
            64,
            1
        )
        .0,
        PITH_E_INVALID
    );
    let vertical: Vec<f64> = vec![2.0, 1.0, 2.0, 5.0, 2.0, 9.0];
    assert_eq!(
        call_alloc_ransac(
            |i, n, t, it, sd, o, l| unsafe { pith_math_ransac_line(i, n, t, it, sd, o, l) },
            &vertical,
            0.5,
            64,
            1
        )
        .0,
        PITH_E_REJECTED
    );
    // complex ops: wrong pair counts are invalid.
    assert_eq!(
        call_alloc(
            |i, n, o, l| unsafe { pith_math_complex_mul(i, n, o, l) },
            &[0.0; 2]
        )
        .0,
        PITH_E_INVALID
    );
    assert_eq!(
        call_alloc(
            |i, n, o, l| unsafe { pith_math_complex_exp(i, n, o, l) },
            &[0.0; 4]
        )
        .0,
        PITH_E_INVALID
    );
    assert_eq!(
        call_alloc_split(
            |i, n, p, o, l| unsafe { pith_math_complex_powi(i, n, p, o, l) },
            &[0.0; 4],
            3
        )
        .0,
        PITH_E_INVALID
    );
    assert_eq!(
        call_scalar(
            |i, n, o| unsafe { pith_math_complex_arg(i, n, o) },
            &[0.0; 4]
        )
        .0,
        PITH_E_INVALID
    );
    // Null pointers stay status codes, never crashes.
    let mut out: *mut f64 = core::ptr::null_mut();
    let mut out_len: usize = 0;
    let mut slot: f64 = 0.0;
    assert_eq!(
        unsafe { pith_math_fft_n(core::ptr::null(), 0, &mut out, &mut out_len) },
        PITH_E_INVALID
    );
    assert_eq!(
        unsafe { pith_math_conv(core::ptr::null(), 0, 0, &mut out, &mut out_len) },
        PITH_E_INVALID
    );
    assert_eq!(
        unsafe { pith_math_lagrange(core::ptr::null(), 0, 1.0, &mut slot) },
        PITH_E_INVALID
    );
    assert_eq!(
        unsafe { pith_math_dtw(core::ptr::null(), 0, 0, &mut slot) },
        PITH_E_INVALID
    );
    assert_eq!(
        unsafe { pith_math_mean(core::ptr::null(), 0, &mut slot) },
        PITH_E_INVALID
    );
    // Null out-slots refuse the same way.
    let data = [1.0, 2.0, 3.0, 4.0];
    assert_eq!(
        unsafe { pith_math_mean(data.as_ptr(), data.len(), core::ptr::null_mut()) },
        PITH_E_INVALID
    );
    assert_eq!(
        unsafe { pith_math_median(data.as_ptr(), data.len(), core::ptr::null_mut()) },
        PITH_E_INVALID
    );
    assert_eq!(
        unsafe { pith_math_complex_arg(data.as_ptr(), 2, core::ptr::null_mut()) },
        PITH_E_INVALID
    );
    // corr with a null input; lagrange with an odd packed length.
    assert_eq!(
        unsafe { pith_math_corr(core::ptr::null(), 0, 0, &mut out, &mut out_len) },
        PITH_E_INVALID
    );
    let odd_pts = [0.0, 1.0, 1.0];
    assert_eq!(
        unsafe { pith_math_lagrange(odd_pts.as_ptr(), odd_pts.len(), 0.5, &mut slot) },
        PITH_E_INVALID
    );
    assert_eq!(
        unsafe { pith_math_complex_powi(core::ptr::null(), 0, 2, &mut out, &mut out_len) },
        PITH_E_INVALID
    );
    assert_eq!(
        unsafe { pith_math_cov(core::ptr::null(), 0, &mut slot) },
        PITH_E_INVALID
    );
    assert_eq!(
        unsafe { pith_math_complex_arg(core::ptr::null(), 0, &mut slot) },
        PITH_E_INVALID
    );
}
