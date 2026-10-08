//! The DCT-III as a first-class transform, 1D and separable 2D.
//!
//! Under the kit's **orthonormal** normalization the DCT-III is
//! exactly the inverse of the DCT-II — the transposed kernel — which
//! [`crate::dct`] already implements as [`crate::idct2`] (and
//! [`crate::idct2_2d`] for the separable 2D form). The tier-1 surface
//! names the transform for what it *is*, not for what it undoes: the
//! same orthonormal DCT-III kernel
//!
//! ```text
//! x[n] = √(2/N)·(F[0]/√2 + Σ_{k≥1} F[k]·cos(π·(2n+1)·k / (2N)))
//! ```
//!
//! reached through one shared implementation rather than a second
//! code path that would have to be re-proven against the definition.
//! [`dct3`]`(`[`dct2`]`(x)) == x` within a few ulps, and the
//! conformance vectors pin both directions.

use crate::dct::{idct2, idct2_2d};

/// 1D orthonormal DCT-III of `f`, returned as a fresh `Vec`.
///
/// `x[n] = √(2/N)·(F[0]/√2 + Σ_{k≥1} F[k]·cos(π·(2n+1)·k/(2N)))` —
/// the exact inverse of [`crate::dct2`], and the same kernel
/// [`crate::idct2`] runs. An empty input returns an empty vector.
#[must_use]
pub fn dct3(f: &[f64]) -> Vec<f64> {
    idct2(f)
}

/// In-place separable 2D orthonormal DCT-III on a `w × h` row-major
/// matrix — the exact inverse of [`crate::dct2_2d`], the same
/// rows-then-columns driver [`crate::idct2_2d`] runs.
///
/// # Panics
///
/// Same shape contract as [`crate::dct2_2d`]: `data.len() == w·h` and
/// positive dimensions.
pub fn dct3_2d(data: &mut [f64], w: usize, h: usize) {
    idct2_2d(data, w, h);
}

#[cfg(test)]
mod tests {
    use super::{dct3, dct3_2d};
    use crate::dct::{dct2, dct2_2d, idct2};

    #[test]
    fn dct3_matches_idct2_bit_for_bit() {
        let x = [0.5, -1.25, 2.0, 3.75, -0.125, 1.5, -2.5, 0.75];
        assert_eq!(dct3(&x), idct2(&x));
    }

    #[test]
    fn dct3_inverts_dct2() {
        let x = [0.5, -1.25, 2.0, 3.75, -0.125, 1.5, -2.5, 0.75];
        let back = dct3(&dct2(&x));
        for (g, w) in back.iter().zip(x.iter()) {
            assert!((g - w).abs() < 1e-14);
        }
    }

    #[test]
    fn dct3_2d_matches_idct2_2d_and_closes_the_round_trip() {
        let orig: Vec<f64> = (0..64).map(|i| ((i * 7) % 11) as f64 - 5.0).collect();
        let mut a = orig.clone();
        let mut b = orig.clone();
        dct3_2d(&mut a, 8, 8);
        crate::dct::idct2_2d(&mut b, 8, 8);
        assert_eq!(a, b);
        dct2_2d(&mut a, 8, 8);
        for (g, w) in a.iter().zip(orig.iter()) {
            assert!((g - w).abs() < 1e-13);
        }
    }
}
