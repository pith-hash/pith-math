//! Interpolation over `f64` samples: the linear blend and the exact
//! polynomial form.
//!
//! [`lerp`] is the affine blend `(1−t)·a + t·b` in the multiply-add
//! form that hits **both** endpoints exactly: `lerp(a, b, 0) == a`
//! and `lerp(a, b, 1) == b` bit-for-bit, the property graphics and
//! resampling code rely on and the naive `a + (b−a)·t` form loses at
//! `t = 1` for many `(a, b)` pairs.
//!
//! [`lagrange_eval`] evaluates the unique polynomial of degree `<
//! n` through `n` points `(x_i, y_i)` at an arbitrary abscissa — the
//! direct O(n²) Lagrange sum
//!
//! ```text
//! p(x) = Σ_i y_i · Π_{j≠i} (x − x_j)/(x_i − x_j)
//! ```
//!
//! the literal definition, chosen for the same reason the DCT kernel
//! is a direct sum: small node counts, and the definition is what the
//! conformance vectors check. Reproducing a node returns that node's
//! `y` exactly; two points sharing an abscissa is not a function
//! specification, and is reported as `None` rather than a divide by
//! zero.

/// The affine blend `(1−t)·a + t·b`, endpoint-exact in the
/// `mul_add` form.
#[must_use]
#[inline]
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    (1.0 - t).mul_add(a, t * b)
}

/// Evaluates the Lagrange interpolant through `points` at `x`.
///
/// Returns `None` when `points` is empty or carries a duplicated
/// abscissa (the interpolant does not exist); a node hit returns that
/// node's ordinate exactly.
#[must_use]
pub fn lagrange_eval(points: &[(f64, f64)], x: f64) -> Option<f64> {
    if points.is_empty() {
        return None;
    }
    let mut acc = 0.0;
    for (i, (xi, yi)) in points.iter().enumerate() {
        // The i-th Lagrange basis polynomial at x.
        let (mut num, mut den) = (1.0, 1.0);
        for (j, (xj, _)) in points.iter().enumerate() {
            if j == i {
                continue;
            }
            num *= x - xj;
            den *= xi - xj;
        }
        if den == 0.0 {
            return None; // duplicated abscissa: not a function.
        }
        acc += yi * (num / den);
    }
    Some(acc)
}

#[cfg(test)]
mod tests {
    use super::{lagrange_eval, lerp};

    #[test]
    fn lerp_hits_both_endpoints_exactly() {
        let (a, b) = (0.1, 1.0e308);
        assert_eq!(lerp(a, b, 0.0), a);
        assert_eq!(lerp(a, b, 1.0), b);
        assert_eq!(lerp(1.0, 3.0, 0.25), 1.5);
    }

    #[test]
    fn lagrange_reproduces_its_nodes_exactly() {
        let pts = [(0.0, 1.0), (1.0, 4.0), (2.0, 9.0)];
        for &(x, y) in &pts {
            assert_eq!(lagrange_eval(&pts, x), Some(y));
        }
    }

    #[test]
    fn lagrange_interpolates_the_quadratic_exactly() {
        // y = x² + 2x + 1 through (0,1),(1,4),(2,9); p(1.5) = 6.25,
        // every basis value dyadic, so the bits are exact.
        let pts = [(0.0, 1.0), (1.0, 4.0), (2.0, 9.0)];
        assert_eq!(lagrange_eval(&pts, 1.5), Some(6.25));
    }

    #[test]
    fn degenerate_node_sets_are_refused() {
        assert_eq!(lagrange_eval(&[], 0.5), None);
        let dupe = [(1.0, 2.0), (1.0, 3.0)];
        assert_eq!(lagrange_eval(&dupe, 0.5), None);
    }

    #[test]
    fn linear_data_interpolates_to_the_line() {
        let pts = [(0.0, 5.0), (4.0, 13.0)]; // y = 2x + 5
        assert_eq!(lagrange_eval(&pts, 2.0), Some(9.0));
    }
}
