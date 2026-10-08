//! Tier-1 expansion acceptance tests: complex arithmetic, the
//! arbitrary-length (Bluestein) DFT, DCT-III, conv/corr, stats,
//! interpolation, RANSAC and DTW.
//!
//! Each kernel is checked against an independently-coded reference —
//! a naive O(N²) definition, a textbook dataset, or a cross-check
//! against a sibling kernel sharing no code path — never against
//! itself. Deterministic randomness stays with `pith_digest`'s
//! `SplitMix64`, the workspace's one PRNG.

use pith_digest::SplitMix64;
use pith_math::{
    Complex, convolve, correlate, covariance, dct2, dct3, dtw_distance, fft, fft_arbitrary,
    ifft_arbitrary, lagrange_eval, lerp, mean, ransac_line, std_dev, variance,
};
use std::f64::consts::{FRAC_1_SQRT_2, PI};

/// Uniform-ish `f64` in `(-1, 1)` from the top 53 bits of `rng`.
fn rand_unit(rng: &mut SplitMix64) -> f64 {
    let bits = (rng.next_u64() >> 11) as f64; // 53-bit mantissa space
    (bits / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// Complex signal with `SplitMix64`-drawn parts.
fn complex_signal(n: usize, seed: u64) -> Vec<Complex> {
    let mut rng = SplitMix64::new(seed);
    (0..n)
        .map(|_| Complex::new(rand_unit(&mut rng), rand_unit(&mut rng)))
        .collect()
}

/// Largest element-wise complex error.
fn max_complex_err(a: &[Complex], b: &[Complex]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x - *y).norm())
        .fold(0.0, f64::max)
}

/// The O(N²) DFT, straight from the definition — the same
/// independently-coded reference the radix-2 tests use.
fn naive_dft(x: &[Complex]) -> Vec<Complex> {
    let n = x.len();
    let mut out = vec![Complex::ZERO; n];
    for (k, slot) in out.iter_mut().enumerate() {
        let mut acc = Complex::ZERO;
        for (j, &xj) in x.iter().enumerate() {
            let w = Complex::cis(-2.0 * PI * (j * k) as f64 / n as f64);
            acc = acc + xj * w;
        }
        *slot = acc;
    }
    out
}

// ---------------------------------------------------------------------
// Complex arithmetic
// ---------------------------------------------------------------------

#[test]
fn euler_identity_exp_i_pi() {
    let e = Complex::new(0.0, PI).exp();
    assert_eq!(e.re, -1.0); // cos(π_f64) rounds to exactly −1.
    assert!(e.im.abs() < 1e-15); // sin(π_f64) is the ulp-scale residue.
}

#[test]
fn arithmetic_identities_close_exactly() {
    let z = Complex::new(3.0, -2.0);
    assert_eq!(z * Complex::new(1.0, -4.0), Complex::new(-5.0, -14.0));
    assert_eq!(z / z, Complex::ONE);
    assert_eq!((z * z.conj()).re, z.norm_sq());
    assert_eq!(z.conj().conj(), z);
    assert_eq!(-z + z, Complex::ZERO);
    // exp/ln are mutual inverses off the cut.
    let w = Complex::new(-1.5, 0.75).ln().exp();
    assert!((w - Complex::new(-1.5, 0.75)).norm() < 1e-15);
}

#[test]
fn sqrt_branches_are_principal() {
    assert_eq!(Complex::new(-4.0, 0.0).sqrt(), Complex::new(0.0, 2.0));
    assert_eq!(Complex::new(-4.0, -0.0).sqrt(), Complex::new(0.0, -2.0));
    let s = Complex::new(0.0, 1.0).sqrt();
    // (1+i)/√2, squared back to i.
    let two = s * s;
    assert!(two.re.is_nan() || (two - Complex::new(0.0, 1.0)).norm() < 1e-15);
    let half = std::f64::consts::FRAC_1_SQRT_2;
    assert!((s.re - half).abs() < 1e-15 && (s.im - half).abs() < 1e-15);
}

#[test]
fn powi_squaring_matches_repeated_multiplication() {
    let z = Complex::new(0.8, -1.3);
    let mut acc = Complex::ONE;
    for p in 1..=9 {
        acc = acc * z;
        assert!((z.powi(p) - acc).norm() < 1e-12, "power {p}");
    }
    // Negative exponents invert.
    let inv = z.powi(-3);
    assert!((inv * z.powi(3) - Complex::ONE).norm() < 1e-12);
}

#[test]
fn division_is_stable_at_large_magnitudes() {
    // The naive form overflows here; Smith's scaling must not.
    let big = Complex::new(1e300, 1e300);
    let q = big / big;
    assert_eq!(q, Complex::ONE);
    let r = Complex::ONE / big;
    assert!(r.re.abs() < 1e-299 && r.im.abs() < 1e-299);
}

#[test]
fn polar_ln_powf_and_the_swapped_division_branches() {
    // from_polar round-trips through arg/norm.
    let z = Complex::from_polar(2.0, core::f64::consts::FRAC_PI_3);
    assert!((z.arg() - core::f64::consts::FRAC_PI_3).abs() < 1e-15);
    assert!((z.norm() - 2.0).abs() < 1e-15);
    assert_eq!(Complex::from_polar(-1.0, 0.0), Complex::new(-1.0, 0.0));

    // Denominator dominated by the imaginary part: 1/i = −i exactly,
    // through recip and Div's swapped branch alike.
    let i = Complex::new(0.0, 1.0);
    assert_eq!(i.recip(), Complex::new(0.0, -1.0));
    assert_eq!(Complex::ONE / i, Complex::new(0.0, -1.0));
    assert_eq!(i * i.recip(), Complex::ONE);

    // sqrt(0) is exactly zero; ln and powf agree with the identities.
    assert_eq!(Complex::ZERO.sqrt(), Complex::ZERO);
    assert_eq!(Complex::new(4.0, 0.0).ln(), Complex::new(4.0_f64.ln(), 0.0));
    // i^2 = −1 through powf's principal branch (e^(2·ln i) = e^(iπ)).
    let sq = Complex::new(0.0, 1.0).powf(2.0);
    assert!(sq.re.is_nan() || (sq - Complex::new(-1.0, 0.0)).norm() < 1e-15);
    // powf agrees with powi for whole exponents on a cut-free value.
    let w = Complex::new(1.2, 0.7);
    let a = w.powf(3.0);
    let b = w.powi(3);
    assert!((a - b).norm() < 1e-12);
    // ln(1) = 0, arg conventions hold on the axes.
    assert_eq!(Complex::ONE.ln(), Complex::ZERO);
    assert_eq!(Complex::new(0.0, 2.0).arg(), core::f64::consts::FRAC_PI_2);
}

// ---------------------------------------------------------------------
// Bluestein — arbitrary-length DFT
// ---------------------------------------------------------------------

#[test]
fn bluestein_matches_naive_dft_on_shared_and_odd_sizes() {
    for n in [2usize, 8, 16, 12, 17, 97] {
        let x = complex_signal(n, 1000 + n as u64);
        let mut got = x.clone();
        fft_arbitrary(&mut got);
        let want = naive_dft(&x);
        // Both kernels share only the libm sin/cos; the error is the
        // twiddle budget, not a bug.
        assert!(
            max_complex_err(&got, &want) < 1e-10,
            "n = {n}: {}",
            max_complex_err(&got, &want)
        );
    }
}

#[test]
fn bluestein_roundtrips_like_the_radix2_kernel() {
    for n in [3usize, 12, 17, 50] {
        let x = complex_signal(n, 2000 + n as u64);
        let mut buf = x.clone();
        fft_arbitrary(&mut buf);
        ifft_arbitrary(&mut buf);
        assert!(max_complex_err(&buf, &x) < 1e-13, "n = {n}");
    }
}

#[test]
fn bluestein_agrees_with_the_radix2_kernel_on_powers_of_two() {
    let x = complex_signal(64, 7);
    let mut fast = x.clone();
    fft_arbitrary(&mut fast);
    let mut radix = x;
    fft(&mut radix);
    assert!(max_complex_err(&fast, &radix) < 1e-10);
}

#[test]
fn bluestein_handles_unit_lengths() {
    let mut one = vec![Complex::new(3.5, -2.0)];
    fft_arbitrary(&mut one);
    assert_eq!(one[0], Complex::new(3.5, -2.0)); // n = 1 is the identity.
    ifft_arbitrary(&mut one); // and the inverse is the same no-op.
    assert_eq!(one[0], Complex::new(3.5, -2.0));
    let mut none: Vec<Complex> = Vec::new();
    fft_arbitrary(&mut none);
    ifft_arbitrary(&mut none);
    assert!(none.is_empty());
}

// ---------------------------------------------------------------------
// DCT-III
// ---------------------------------------------------------------------

/// The orthonormal DCT-III straight from the definition.
fn naive_dct3(f: &[f64]) -> Vec<f64> {
    let n = f.len();
    (0..n)
        .map(|n_i| {
            let sum: f64 = f
                .iter()
                .enumerate()
                .map(|(k, &fk)| {
                    let c = if k == 0 { FRAC_1_SQRT_2 } else { 1.0 };
                    c * fk * (PI * (2 * n_i + 1) as f64 * k as f64 / (2.0 * n as f64)).cos()
                })
                .sum::<f64>();
            (2.0 / n as f64).sqrt() * sum
        })
        .collect()
}

#[test]
fn dct3_matches_the_direct_definition() {
    let f: Vec<f64> = (0..16).map(|i| ((i * 7) % 13) as f64 - 6.0).collect();
    let got = dct3(&f);
    let want = naive_dct3(&f);
    for (g, w) in got.iter().zip(want.iter()) {
        assert!((g - w).abs() < 1e-12);
    }
}

#[test]
fn dct3_inverts_dct2_both_ways() {
    let x: Vec<f64> = (0..32)
        .map(|i| ((i * 37) % 32) as f64 * 0.125 - 2.0)
        .collect();
    let back = dct3(&dct2(&x));
    for (g, w) in back.iter().zip(x.iter()) {
        assert!((g - w).abs() < 1e-13);
    }
}

// ---------------------------------------------------------------------
// conv / corr
// ---------------------------------------------------------------------

/// The direct O(n·m) sliding sum — the definition.
fn naive_convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, slot) in out.iter_mut().enumerate() {
        for j in 0..a.len() {
            if i >= j && i - j < b.len() {
                *slot += a[j] * b[i - j];
            }
        }
    }
    out
}

#[test]
fn convolve_known_small_cases_are_exact() {
    assert_eq!(
        convolve(&[1.0, 2.0, 3.0], &[0.0, 1.0, 2.0]),
        vec![0.0, 1.0, 4.0, 7.0, 6.0]
    );
    assert_eq!(convolve(&[2.0], &[7.0]), vec![14.0]);
    assert!(convolve(&[], &[1.0]).is_empty());
}

#[test]
fn fft_path_agrees_with_the_direct_definition() {
    let mut rng = SplitMix64::new(42);
    let a: Vec<f64> = (0..150).map(|_| rand_unit(&mut rng)).collect();
    let b: Vec<f64> = (0..97).map(|_| rand_unit(&mut rng)).collect();
    let want = naive_convolve(&a, &b);
    let got = convolve(&a, &b);
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want.iter()) {
        assert!((g - w).abs() < 1e-9, "|{g} − {w}|");
    }
}

#[test]
fn correlate_is_convolution_with_the_flipped_kernel() {
    assert_eq!(correlate(&[1.0, 2.0], &[3.0, 4.0]), vec![4.0, 11.0, 6.0]);
    let mut rng = SplitMix64::new(43);
    let a: Vec<f64> = (0..60).map(|_| rand_unit(&mut rng)).collect();
    let b: Vec<f64> = (0..31).map(|_| rand_unit(&mut rng)).collect();
    let flipped = convolve(&a, &b.iter().rev().copied().collect::<Vec<f64>>());
    let got = correlate(&a, &b);
    for (g, w) in got.iter().zip(flipped.iter()) {
        assert!((g - w).abs() < 1e-9);
    }
}

#[test]
fn convolution_with_the_unit_impulse_is_the_identity() {
    let x: Vec<f64> = (0..100).map(|i| ((i * 17) % 9) as f64 - 4.0).collect();
    let got = convolve(&x, &[1.0]);
    assert_eq!(got, x); // exact: the direct path multiplies by a lone 1.
    // A zero impulse annihilates exactly.
    assert!(convolve(&x, &[0.0]).iter().all(|&v| v == 0.0));
}

// ---------------------------------------------------------------------
// stats
// ---------------------------------------------------------------------

#[test]
fn textbook_datasets_land_on_exact_rationals() {
    assert_eq!(mean(&[1.0, 2.0, 3.0, 4.0]), Some(2.5));
    assert_eq!(variance(&[1.0, 3.0]), Some(2.0));
    assert_eq!(covariance(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]), Some(2.0));
}

#[test]
fn variance_matches_the_two_pass_definition_on_random_data() {
    let mut rng = SplitMix64::new(11);
    let x: Vec<f64> = (0..64).map(|_| rand_unit(&mut rng) * 10.0).collect();
    let mu = mean(&x).unwrap();
    let want: f64 = x.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / 63.0;
    assert!((variance(&x).unwrap() - want).abs() < 1e-12);
    assert!((std_dev(&x).unwrap() - want.sqrt()).abs() < 1e-12);
}

#[test]
fn covariance_of_independent_coordinates_centers_on_zero() {
    let mut rng = SplitMix64::new(12);
    let a: Vec<f64> = (0..256).map(|_| rand_unit(&mut rng)).collect();
    let b: Vec<f64> = (0..256).map(|_| rand_unit(&mut rng)).collect();
    let cov = covariance(&a, &b).unwrap();
    assert!(cov.abs() < 0.2, "independent draws: cov = {cov}");
    // Perfectly correlated data saturates at the variance product.
    assert!((covariance(&a, &a).unwrap() - variance(&a).unwrap()).abs() < 1e-9);
}

#[test]
fn stats_refusals_follow_the_median_convention() {
    assert_eq!(mean(&[]), None);
    assert_eq!(variance(&[5.0]), None);
    assert_eq!(covariance(&[1.0], &[1.0, 2.0]), None);
}

// ---------------------------------------------------------------------
// interp
// ---------------------------------------------------------------------

#[test]
fn lerp_hits_both_endpoints_exactly() {
    let (a, b) = (0.3, 5.0e307);
    assert_eq!(lerp(a, b, 0.0), a);
    assert_eq!(lerp(a, b, 1.0), b);
    assert_eq!(lerp(1.0, 3.0, 0.25), 1.5);
}

#[test]
fn lagrange_reproduces_a_random_polynomial_everywhere() {
    // Degree-4 polynomial through 5 integer nodes, sampled off-node.
    let coeffs = [3.0, -2.0, 0.5, 1.0, -0.25];
    let poly = |x: f64| {
        coeffs
            .iter()
            .rev()
            .copied()
            .reduce(|acc, c| acc * x + c)
            .unwrap()
    };
    let nodes: Vec<(f64, f64)> = (0..5).map(|i| (i as f64, poly(i as f64))).collect();
    for x in [0.5, 1.25, 2.75, 4.5, -1.0] {
        let want = poly(x);
        let got = lagrange_eval(&nodes, x).unwrap();
        assert!((got - want).abs() < 1e-9 * want.abs().max(1.0), "x = {x}");
    }
}

#[test]
fn lagrange_refuses_degenerate_node_sets() {
    assert_eq!(lagrange_eval(&[], 1.0), None);
    assert_eq!(lagrange_eval(&[(1.0, 2.0), (1.0, 3.0)], 0.5), None);
}

// ---------------------------------------------------------------------
// RANSAC
// ---------------------------------------------------------------------

#[test]
fn ransac_recovers_a_planted_line_through_outliers() {
    // 30 points on y = −0.75x + 2 (dyadic), 6 gross outliers.
    let mut pts: Vec<(f64, f64)> = (0..30)
        .map(|i| {
            let x = i as f64 * 0.5;
            (x, -0.75 * x + 2.0)
        })
        .collect();
    pts.push((3.0, 50.0));
    pts.push((6.5, -40.0));
    pts.push((9.0, 33.0));
    pts.push((1.0, -25.0));
    pts.push((12.0, 18.0));
    pts.push((0.5, -9.0));
    let fit = ransac_line(&pts, 1e-9, 256, 2026).expect("planted line found");
    assert_eq!(fit.slope, -0.75);
    assert_eq!(fit.intercept, 2.0);
    assert_eq!(fit.inliers, 30);
}

#[test]
fn ransac_seeded_runs_replay_bit_for_bit() {
    let mut rng = SplitMix64::new(99);
    let pts: Vec<(f64, f64)> = (0..80)
        .map(|i| {
            let jitter = rand_unit(&mut rng) * 0.01;
            (i as f64, 3.0 * i as f64 - 1.0 + jitter)
        })
        .collect();
    let a = ransac_line(&pts, 0.02, 200, 7);
    let b = ransac_line(&pts, 0.02, 200, 7);
    assert_eq!(a, b);
    let c = ransac_line(&pts, 0.02, 200, 8); // different seed, valid model
    assert!(c.is_some());
}

#[test]
fn ransac_refuses_degenerate_configurations() {
    let pts = [(0.0, 0.0), (1.0, 1.0)];
    assert_eq!(ransac_line(&pts[..1], 0.5, 32, 1), None);
    assert_eq!(ransac_line(&pts, 0.5, 0, 1), None);
    assert_eq!(ransac_line(&pts, 0.0, 32, 1), None);
    assert_eq!(ransac_line(&pts, -0.5, 32, 1), None);
    let vertical = [(2.0, 0.0), (2.0, 1.0), (2.0, 2.0)];
    assert_eq!(ransac_line(&vertical, 0.5, 32, 1), None);
}

// ---------------------------------------------------------------------
// DTW
// ---------------------------------------------------------------------

#[test]
fn dtw_textbook_cost_matrices_are_reproduced() {
    assert_eq!(dtw_distance(&[1.0, 2.0, 3.0], &[2.0, 2.0, 2.0]), Some(2.0));
    assert_eq!(dtw_distance(&[1.0, 3.0], &[2.0, 2.0, 4.0]), Some(3.0));
    assert_eq!(
        dtw_distance(&[1.0, 2.0, 3.0], &[1.0, 2.0, 2.0, 3.0]),
        Some(0.0)
    );
}

#[test]
fn dtw_matches_a_direct_memoized_reference() {
    // Independent top-down implementation with an explicit memo table.
    fn rec(a: &[f64], b: &[f64], i: usize, j: usize, memo: &mut Vec<Vec<Option<f64>>>) -> f64 {
        if i == 0 && j == 0 {
            return (a[0] - b[0]).abs();
        }
        if let Some(v) = memo[i][j] {
            return v;
        }
        let local = (a[i] - b[j]).abs();
        let mut best = f64::INFINITY;
        if i > 0 {
            best = best.min(rec(a, b, i - 1, j, memo));
        }
        if j > 0 {
            best = best.min(rec(a, b, i, j - 1, memo));
        }
        if i > 0 && j > 0 {
            best = best.min(rec(a, b, i - 1, j - 1, memo));
        }
        let v = local + best;
        memo[i][j] = Some(v);
        v
    }
    let mut rng = SplitMix64::new(17);
    let a: Vec<f64> = (0..23).map(|_| rand_unit(&mut rng)).collect();
    let b: Vec<f64> = (0..31).map(|_| rand_unit(&mut rng)).collect();
    let mut memo = vec![vec![None; b.len()]; a.len()];
    let want = rec(&a, &b, a.len() - 1, b.len() - 1, &mut memo);
    assert!((dtw_distance(&a, &b).unwrap() - want).abs() < 1e-9);
}

#[test]
fn dtw_refuses_empty_sequences() {
    assert_eq!(dtw_distance(&[], &[1.0]), None);
    assert_eq!(dtw_distance(&[1.0], &[]), None);
}
