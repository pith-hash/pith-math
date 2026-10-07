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
    PITH_E_INVALID, PITH_E_REJECTED, PITH_OK, pith_math_dct2, pith_math_dct2_2d, pith_math_det3,
    pith_math_fft, pith_math_fft_real, pith_math_free, pith_math_idct2, pith_math_ifft,
    pith_math_inverse3, pith_math_mat3_mul, pith_math_mat3_mul_vec, pith_math_median,
    pith_math_solve3, pith_math_transpose3,
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
