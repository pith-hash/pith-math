//! The Bluestein chirp-z DFT: the transform of **any** length `n ≥ 1`,
//! not only powers of two.
//!
//! [`crate::fft`] is the iterative radix-2 kernel — exact and fast,
//! but `n` must be a power of two, and the kit's real signals (prime
//! window sizes, sub-window pitches, resampled blocks) are not. The
//! classic fix rewrites the DFT sum into a convolution through the
//! chirp identity `2jk = j² + k² − (k−j)²`:
//!
//! ```text
//! X[k] = c[k] · Σ_j (x[j]·c[j]) · conj(c[(k−j) mod 2n]),
//! c[t] = e^(−iπ·t²/n)
//! ```
//!
//! a linear convolution of the pre-chirped input with the (even,
//! real-symmetric-in-index) chirp — computable with the existing
//! radix-2 [`crate::fft`] zero-padded to `m ≥ 2n − 1` (the next power
//! of two). The result is an O(m log m) transform for every length,
//! sharing the one proven butterfly kernel.
//!
//! Accuracy matches the radix-2 kernel: every twiddle comes from a
//! fresh [`Complex::cis`], and the chirp angle is reduced through
//! `t² mod 2n` *before* the `sin`/`cos`, so precision never depends on
//! `n` through the size of `t²` — the accumulated-phase trap of naive
//! chirp implementations.
//!
//! [`ifft_arbitrary`] is the conjugate trick again:
//! `conj(fft_arbitrary(conj(X)))/n`, so the forward kernel stays the
//! only code path.

use std::f64::consts::PI;

use crate::complex::Complex;
use crate::fft::{fft, ifft};

/// In-place DFT of **any** length `n ≥ 1`:
/// `X[k] = Σ_j x[j]·e^(−2πi·j·k/N)` — the same unnormalized physics
/// convention as [`crate::fft`], lifted power-of-two restriction.
///
/// `n = 0` is a no-op; `n = 1` is the identity. Every other length
/// runs the chirp-z convolution above on a scratch buffer of the next
/// power of two `≥ 2n − 1`.
///
/// # Panics
///
/// Panics only if `n` exceeds `2³²` (the chirp index reduction squares
/// indices in a `u64`); the kit's signals are nowhere near that.
pub fn fft_arbitrary(buf: &mut [Complex]) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    // The convolution length: a power of two covering both supports.
    let m = (2 * n - 1).next_power_of_two();
    // Chirp `c[t] = e^(−iπ·t²/n)` with the angle reduced through
    // `t² mod 2n` — the chirp has period 2n in t, and t ≤ n here.
    let chirp = |t: usize| -> Complex {
        let red = ((t as u64 * t as u64) % (2 * n as u64)) as f64;
        Complex::cis(-PI * red / (n as f64))
    };

    // a = x·c, zero-padded to the convolution length.
    let mut a: Vec<Complex> = (0..n).map(|j| buf[j] * chirp(j)).collect();
    a.resize(m, Complex::ZERO);
    // b = the even chirp w[t] = conj(c[t]): w[t] at t ∈ [0, n) and its
    // mirror w[m − t] = w[−t] at the wrapped tail — exactly how a
    // length-m circular convolution carries the negative indices.
    let mut b = vec![Complex::ZERO; m];
    for t in 0..n {
        let w = chirp(t).conj();
        b[t] = w;
        if t > 0 {
            b[m - t] = w;
        }
    }

    fft(&mut a);
    fft(&mut b);
    for (x, w) in a.iter_mut().zip(b.iter()) {
        *x = *x * *w;
    }
    ifft(&mut a);

    // Post-chirp, keeping only the first n bins.
    for (k, slot) in buf.iter_mut().enumerate() {
        *slot = a[k] * chirp(k);
    }
}

/// In-place inverse DFT of any length — the exact (within ulps)
/// inverse of [`fft_arbitrary`], same conjugate trick as
/// [`crate::ifft`].
///
/// # Panics
///
/// Same contract as [`fft_arbitrary`].
pub fn ifft_arbitrary(buf: &mut [Complex]) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    for x in buf.iter_mut() {
        *x = x.conj();
    }
    fft_arbitrary(buf);
    let inv = 1.0 / (n as f64);
    for x in buf.iter_mut() {
        *x = x.conj().scale(inv);
    }
}
