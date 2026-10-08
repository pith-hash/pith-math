//! Linear convolution and cross-correlation over `f64`.
//!
//! [`convolve`] answers "what does filter `b` do to signal `a`":
//! `(a ∗ b)[i] = Σ_k a[k]·b[i−k]`, full support `n + m − 1`. Two
//! engines behind one signature:
//!
//! * the **direct** O(n·m) sliding sum for small inputs — literally
//!   the definition, exact on integer data, the path every conformance
//!   vector exercises;
//! * the **FFT path** once the work product `n·m` crosses a threshold:
//!   transform both operands with the arbitrary-length
//!   [`fft_arbitrary`], multiply the spectra, invert, truncate to the
//!   full-support length. The convolution theorem holds for *any*
//!   length here because [`fft_arbitrary`] has no power-of-two
//!   constraint — no padding gymnastics at the call site.
//!
//! [`correlate`] is convolution against the reversed second operand —
//! the standard full-support cross-correlation
//! `(a ⋆ b)[k] = Σ_j a[j+k−(m−1)]·b[j]` — so it inherits both engines
//! and one proven kernel.

use crate::bluestein::{fft_arbitrary, ifft_arbitrary};
use crate::complex::Complex;

/// The `n·m` product above which the FFT path replaces the direct
/// sliding sum: at 4096 multiply-adds the transform overhead is still
/// noise, and below it the direct sum is both faster at these sizes
/// and the exact definition (integer inputs stay integer-exact).
const DIRECT_WORK_LIMIT: usize = 4096;

/// Full-support linear convolution `(a ∗ b)[i] = Σ_k a[k]·b[i−k]`,
/// returned as a fresh `Vec` of `a.len() + b.len() − 1` samples.
///
/// An empty operand convolves to nothing — an empty result, matching
/// the degenerate support arithmetic rather than an error.
pub fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let work = a.len().saturating_mul(b.len());
    if work <= DIRECT_WORK_LIMIT {
        convolve_direct(a, b)
    } else {
        convolve_fft(a, b)
    }
}

/// Full-support cross-correlation `(a ⋆ b)[k] = Σ_j a[j+k−(m−1)]·b[j]`
/// over `a.len() + b.len() − 1` lags — convolution with the reversed
/// `b`, the identity the whole discipline is built on.
pub fn correlate(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut flipped = b.to_vec();
    flipped.reverse();
    convolve(a, &flipped)
}

/// The definition, straight out: for every output index sum the
/// in-bounds products. O(n·m), no allocation beyond the result.
fn convolve_direct(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len() + b.len() - 1;
    let mut out = vec![0.0; n];
    for (i, slot) in out.iter_mut().enumerate() {
        // j indexes a, i − j indexes b; both must stay in bounds.
        let lo = i.saturating_sub(b.len() - 1);
        let hi = i.min(a.len() - 1);
        let mut acc = 0.0;
        for j in lo..=hi {
            acc += a[j] * b[i - j];
        }
        *slot = acc;
    }
    out
}

/// The convolution theorem path: `ifft(fft(a)·fft(b))` on a common
/// zero-padded length, truncated to the full-support `n + m − 1`.
fn convolve_fft(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len() + b.len() - 1;
    let m = n.next_power_of_two();
    let mut fa: Vec<Complex> = a.iter().map(|&re| Complex::new(re, 0.0)).collect();
    fa.resize(m, Complex::ZERO);
    let mut fb: Vec<Complex> = b.iter().map(|&re| Complex::new(re, 0.0)).collect();
    fb.resize(m, Complex::ZERO);
    fft_arbitrary(&mut fa);
    fft_arbitrary(&mut fb);
    for (x, w) in fa.iter_mut().zip(fb.iter()) {
        *x = *x * *w;
    }
    ifft_arbitrary(&mut fa);
    fa.truncate(n);
    // The convolution of real signals is real; the tiny imaginary
    // residue is transform noise, discarded.
    fa.iter().map(|c| c.re).collect()
}

#[cfg(test)]
mod tests {
    use super::{convolve, correlate};

    #[test]
    fn convolve_matches_the_definition_on_known_small_cases() {
        assert_eq!(
            convolve(&[1.0, 2.0, 3.0], &[0.0, 1.0, 2.0]),
            vec![0.0, 1.0, 4.0, 7.0, 6.0]
        );
        assert_eq!(convolve(&[1.0], &[5.0]), vec![5.0]);
    }

    #[test]
    fn empty_operand_convolves_to_nothing() {
        assert!(convolve(&[], &[1.0, 2.0]).is_empty());
        assert!(convolve(&[1.0, 2.0], &[]).is_empty());
    }

    #[test]
    fn fft_path_agrees_with_the_direct_definition() {
        let a: Vec<f64> = (0..97).map(|i| ((i * 13) % 7) as f64 - 3.0).collect();
        let b: Vec<f64> = (0..41).map(|i| ((i * 29) % 11) as f64 - 5.0).collect();
        let direct = super::convolve_direct(&a, &b);
        let fast = convolve(&a, &b);
        assert_eq!(fast.len(), direct.len());
        for (g, w) in fast.iter().zip(direct.iter()) {
            assert!((g - w).abs() < 1e-9, "|{g} − {w}| = {}", (g - w).abs());
        }
    }

    #[test]
    fn correlate_is_convolution_with_the_flipped_kernel() {
        assert_eq!(correlate(&[1.0, 2.0], &[3.0, 4.0]), vec![4.0, 11.0, 6.0]);
        // Self-correlation peaks at the centred zero lag.
        let x = [1.0, -2.0, 3.0, 4.0];
        let ac = correlate(&x, &x);
        let peak = ac
            .iter()
            .enumerate()
            .max_by(|(_, p), (_, q)| p.total_cmp(q))
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(peak, x.len() - 1);
    }
}
