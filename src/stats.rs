//! Descriptive statistics over `f64` slices: mean, variance,
//! covariance, standard deviation.
//!
//! Every estimator here is the **sample** form — the `n − 1`
//! denominator (Bessel's correction) — because the kit's consumers
//! treat a slice as a sample of a signal, not the whole population,
//! and the unbiased form is the one RANSAC refinement and spectral
//! statistics want. The population form is one line away
//! (`variance(x)·(n−1)/n`) and deliberately not a second API.
//!
//! Empty inputs return `None` (the median convention); variance and
//! covariance additionally need `n ≥ 2` — the `n − 1` denominator is
//! not a number for a single observation. Numeric form is the plain
//! two-pass textbook estimator (mean first, then the summed squared
//! deviations): for the kit's magnitudes it is stable, exact on
//! representable rationals, and *is* the definition the conformance
//! vectors check.

/// The arithmetic mean `Σx/n`, or `None` on an empty slice.
#[must_use]
pub fn mean(x: &[f64]) -> Option<f64> {
    if x.is_empty() {
        return None;
    }
    let sum: f64 = x.iter().sum();
    Some(sum / x.len() as f64)
}

/// The sample variance `Σ(x−x̄)²/(n−1)`, or `None` for fewer than two
/// observations.
#[must_use]
pub fn variance(x: &[f64]) -> Option<f64> {
    let mu = mean(x)?;
    if x.len() < 2 {
        return None;
    }
    Some(squared_deviation(x, mu) / (x.len() - 1) as f64)
}

/// The sample standard deviation, the square root of
/// [`variance`] — `None` under the same conditions.
#[must_use]
pub fn std_dev(x: &[f64]) -> Option<f64> {
    variance(x).map(f64::sqrt)
}

/// The sample covariance `Σ(a−ā)(b−b̄)/(n−1)` over two equal-length
/// series — `None` on empty inputs, a length mismatch, or fewer than
/// two paired observations.
#[must_use]
pub fn covariance(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.is_empty() || a.len() != b.len() || a.len() < 2 {
        return None;
    }
    let ma = mean(a)?;
    let mb = mean(b)?;
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - ma) * (y - mb))
        .sum();
    Some(sum / (a.len() - 1) as f64)
}

/// Total squared deviation from `mu`, shared by the variance estimators.
fn squared_deviation(x: &[f64], mu: f64) -> f64 {
    x.iter().map(|&v| (v - mu) * (v - mu)).sum()
}

#[cfg(test)]
mod tests {
    use super::{covariance, mean, std_dev, variance};

    #[test]
    fn textbook_datasets_land_on_exact_rationals() {
        assert_eq!(mean(&[1.0, 2.0, 3.0, 4.0]), Some(2.5));
        assert_eq!(mean(&[7.0]), Some(7.0));
        // [1,3]: deviations ±1, sample variance 2 exactly.
        assert_eq!(variance(&[1.0, 3.0]), Some(2.0));
        // Paired (1,2),(2,4),(3,6): deviations (−1,−2),(0,0),(1,2),
        // cross products 2+0+2 over n−1 = 2 → exactly 2.
        assert_eq!(covariance(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]), Some(2.0));
    }

    #[test]
    fn non_representable_results_land_within_a_few_ulps() {
        // 5/3 is not a binary rational; the bits are shaped by the
        // division, the value is not.
        let v = variance(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!((v - 5.0 / 3.0).abs() < 1e-15);
    }

    #[test]
    fn empty_and_singleton_refusals() {
        assert_eq!(mean(&[]), None);
        assert_eq!(variance(&[]), None);
        assert_eq!(variance(&[1.0]), None);
        assert_eq!(covariance(&[1.0], &[1.0, 2.0]), None);
        assert_eq!(covariance(&[], &[]), None);
    }

    #[test]
    fn std_dev_is_the_square_root_of_variance() {
        let x = [1.0, 3.0];
        assert_eq!(std_dev(&x), Some(2.0_f64.sqrt()));
    }

    #[test]
    fn covariance_is_symmetric_and_scale_tracks_linear_maps() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [10.0, 20.0, 30.0, 50.0];
        let ab = covariance(&a, &b).unwrap();
        let ba = covariance(&b, &a).unwrap();
        assert!((ab - ba).abs() < 1e-12);
        // A perfect linear relation carries the slope in the
        // covariance: b = 5a + c ⇒ cov(a,b) = 5·var(a).
        let perfect = covariance(&a, &a.iter().map(|v| 5.0 * v + 1.0).collect::<Vec<_>>()).unwrap();
        let va = variance(&a).unwrap();
        assert!((perfect - 5.0 * va).abs() < 1e-9);
    }
}
