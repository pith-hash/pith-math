//! Dynamic time warping over `f64` sequences.
//!
//! DTW aligns two sequences that may run at different speeds by
//! finding the monotone warping path that minimizes the total local
//! cost — the classic speech/audio distance and the alignment half of
//! every motif-matching pipeline. Local cost is the absolute
//! difference `|a[i] − b[j]|` (the 1-D form of the usual L¹ frame
//! cost); allowed steps are the three monotone moves down, right,
//! diagonal. The dynamic program
//!
//! ```text
//! D[i][j] = |a[i] − b[j]| + min(D[i−1][j], D[i][j−1], D[i−1][j−1])
//! ```
//!
//! runs over a `(n+1)×(m+1)` grid with an infinity border and keeps
//! only two rows — O(n·m) time, O(min(n, m)) extra space.
//!
//! The conformance vectors are the textbook integer cost matrices
//! (e.g. the 3×3 and 2×3 examples from the DTW literature), whose
//! optimal path costs are small integers — bit-exact through this
//! code path.

/// The DTW distance between `a` and `b` — the minimal total `|a − b|`
/// cost of any monotone alignment path — or `None` when either
/// sequence is empty.
#[must_use]
pub fn dtw_distance(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    // Iterate the shorter sequence across (inner loop) so the two
    // rolling rows stay as small as the geometry allows.
    let (rows, cols) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let inf = f64::INFINITY;
    let mut prev = vec![inf; cols.len() + 1];
    let mut curr = vec![inf; cols.len() + 1];
    prev[0] = 0.0;
    for &r in rows.iter() {
        curr[0] = inf;
        for (j, &c) in cols.iter().enumerate() {
            let local = (r - c).abs();
            let best_prev = prev[j + 1].min(curr[j]).min(prev[j]);
            curr[j + 1] = local + best_prev;
        }
        core::mem::swap(&mut prev, &mut curr);
    }
    Some(prev[cols.len()])
}

#[cfg(test)]
mod tests {
    use super::dtw_distance;

    #[test]
    fn textbook_cost_matrices_reproduce_integer_optima() {
        // X=(1,2,3) vs Y=(2,2,2): local costs {1,0,1}, optimal path 1+0+1.
        assert_eq!(dtw_distance(&[1.0, 2.0, 3.0], &[2.0, 2.0, 2.0]), Some(2.0));
        // (1,3) vs (2,2,4): every alignment of the 3 costs 1 somewhere;
        // the optimum is 1+1+1 along 1→2, 3→2, 3→4.
        assert_eq!(dtw_distance(&[1.0, 3.0], &[2.0, 2.0, 4.0]), Some(3.0));
    }

    #[test]
    fn identical_sequences_cost_zero() {
        let x = [3.0, -1.0, 4.5];
        assert_eq!(dtw_distance(&x, &x), Some(0.0));
    }

    #[test]
    fn distance_is_symmetric() {
        let a = [1.0, 2.0, 5.0, 3.0];
        let b = [2.0, 2.0, 2.0, 2.0, 4.0];
        let ab = dtw_distance(&a, &b).unwrap();
        let ba = dtw_distance(&b, &a).unwrap();
        assert_eq!(ab, ba);
    }

    #[test]
    fn empty_sequences_are_none() {
        assert_eq!(dtw_distance(&[], &[1.0]), None);
        assert_eq!(dtw_distance(&[1.0], &[]), None);
        assert_eq!(dtw_distance(&[], &[]), None);
    }

    #[test]
    fn a_repeated_middle_sample_is_absorbed_for_free() {
        // The tempo-stretched twin of [1,2,3] warps onto it at zero
        // cost — the alignment DTW exists for, and one no lockstep
        // distance can express.
        assert_eq!(
            dtw_distance(&[1.0, 2.0, 3.0], &[1.0, 2.0, 2.0, 3.0]),
            Some(0.0)
        );
    }
}
