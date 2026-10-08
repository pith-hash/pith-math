//! RANSAC (RANdom SAmple Consensus) over `f64` data — the
//! general-purpose robust estimator the tier-1 surface exposes as a
//! line fit.
//!
//! RANSAC rides out outliers by fitting minimal models to random
//! samples and keeping the model with the largest consensus set. The
//! canonical loop, implemented here exactly:
//!
//! 1. draw an unordered pair of distinct points (rejection sampling
//!    from the seeded [`SplitMix64`]);
//! 2. skip *degenerate* samples — equal abscissae have no line;
//! 3. count the inliers of the through-pair line under `threshold`;
//! 4. keep the first model that beats the running best.
//!
//! Determinism is a contract, not a hope: the random stream is the
//! fleet's [`SplitMix64`] (the same generator `pith-digest` ships,
//! seeded — never the clock), the scan order is fixed, and ties go to
//! the first model found. Every intermediate quantity is plain
//! `f64` multiply/divide on the caller's coordinates — no `mul_add`,
//! no platform variance — so a seeded run replays bit-for-bit
//! anywhere, which is what the exact conformance vector pins.
//!
//! The best **hypothesis** is returned, not a refit: classic RANSAC,
//! one decision fewer, and the consensus set is available to callers
//! that want to refine.

/// The splitmix64 generator (Steele, Lea & Flood 2014) — a direct
/// port of `pith-digest`'s, kept here because the library proper is
/// dependency-free and the seeded RANSAC stream must live inside it.
///
/// Advances by the golden gamma and runs the xorshift-multiply output
/// function. Tests and conformance vectors seed it with a constant,
/// never with the clock.
#[derive(Clone)]
pub struct SplitMix64 {
    /// The 64-bit internal state.
    state: u64,
}

impl SplitMix64 {
    /// Creates a generator seeded with `seed`.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    /// Produces the next 64-bit output and advances the state.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// A fitted line `y = slope·x + intercept` and its consensus size.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LineFit {
    /// The slope of the fitted line.
    pub slope: f64,
    /// The intercept of the fitted line.
    pub intercept: f64,
    /// Number of points within `threshold` of the line.
    pub inliers: usize,
}

/// RANSAC line fit over `(x, y)` points.
///
/// Runs `iterations` random two-point samples under the inlier
/// `threshold` and returns the model with the largest consensus set,
/// or `None` when the configuration is degenerate: fewer than two
/// points, a zero or negative `threshold` (an empty consensus is
/// meaningless), zero iterations — or every sample drawn turned out
/// vertical, so no model existed to score.
///
/// An unordered pair is drawn by rejection: two independent indices,
/// redrawn while equal (bounded retries, then the iteration is
/// skipped as degenerate).
#[must_use]
pub fn ransac_line(
    points: &[(f64, f64)],
    threshold: f64,
    iterations: usize,
    seed: u64,
) -> Option<LineFit> {
    if points.len() < 2 || iterations == 0 || threshold.is_nan() || threshold <= 0.0 {
        return None;
    }
    let n = points.len() as u64;
    let mut rng = SplitMix64::new(seed);
    let mut best: Option<LineFit> = None;
    for _ in 0..iterations {
        // Rejection-sample an unordered index pair.
        let i = (rng.next_u64() % n) as usize;
        let mut j = (rng.next_u64() % n) as usize;
        let mut tries = 0;
        while j == i {
            j = (rng.next_u64() % n) as usize;
            tries += 1;
            if tries > 32 {
                break;
            }
        }
        if j == i {
            continue;
        }
        let (p, q) = (points[i], points[j]);
        if p.0 == q.0 {
            continue; // vertical: degenerate for a function model.
        }
        let slope = (q.1 - p.1) / (q.0 - p.0);
        let intercept = p.1 - slope * p.0;
        // Plain product form, no `mul_add`: FMA availability is
        // platform-shaped, and the seeded stream must replay
        // bit-for-bit everywhere (see the module note).
        let inliers = points
            .iter()
            .filter(|&&(x, y)| (y - (slope * x + intercept)).abs() <= threshold)
            .count();
        if best.is_none_or(|b| inliers > b.inliers) {
            best = Some(LineFit {
                slope,
                intercept,
                inliers,
            });
        }
    }
    best
}

/// The exactness of the seeded stream, pinned: the port must match
/// `pith-digest`'s output word for word.
#[cfg(test)]
mod tests {
    use super::{SplitMix64, ransac_line};

    #[test]
    fn splitmix64_matches_the_fleet_stream() {
        let mut rng = SplitMix64::new(0);
        assert_eq!(rng.next_u64(), 0xe220_a839_7b1d_cdaf);
        assert_eq!(rng.next_u64(), 0x6e78_9e6a_a1b9_65f4);
    }

    #[test]
    fn recovers_an_exact_line_through_noise() {
        // Four points on y = 0.5x + 1 plus two gross outliers.
        let pts = [
            (0.0, 1.0),
            (2.0, 2.0),
            (4.0, 3.0),
            (6.0, 4.0),
            (2.0, 10.0),
            (4.0, -9.0),
        ];
        let fit = ransac_line(&pts, 0.5, 64, 42).expect("a dominant line exists");
        assert_eq!(fit.slope, 0.5);
        assert_eq!(fit.intercept, 1.0);
        assert_eq!(fit.inliers, 4);
    }

    #[test]
    fn seeded_runs_replay_bit_for_bit() {
        let pts: Vec<(f64, f64)> = (0..40)
            .map(|i| {
                let mut rng = SplitMix64::new(i as u64);
                let jitter = (rng.next_u64() % 1000) as f64 / 8000.0;
                (i as f64, 2.0 * i as f64 + 1.0 + jitter)
            })
            .collect();
        let a = ransac_line(&pts, 0.05, 128, 7);
        let b = ransac_line(&pts, 0.05, 128, 7);
        assert_eq!(a, b);
    }

    #[test]
    fn degenerate_configurations_are_none() {
        let pts = [(0.0, 0.0), (1.0, 1.0)];
        assert_eq!(ransac_line(&pts[..1], 0.5, 32, 1), None, "single point");
        assert_eq!(ransac_line(&[], 0.5, 32, 1), None, "empty");
        assert_eq!(ransac_line(&pts, 0.5, 0, 1), None, "no iterations");
        assert_eq!(ransac_line(&pts, 0.0, 32, 1), None, "zero threshold");
        assert_eq!(ransac_line(&pts, -1.0, 32, 1), None, "negative threshold");
    }

    #[test]
    fn all_vertical_points_yield_no_model() {
        let pts = [(2.0, 1.0), (2.0, 5.0), (2.0, 9.0)];
        assert_eq!(ransac_line(&pts, 0.5, 16, 3), None);
    }
}
