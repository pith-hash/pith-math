//! The numeric kernel of the `pith` kit: Fourier transforms, the
//! DCT, order statistics, and the smallest linear algebra that tier 3
//! needs — plus the tier-1 expansion: full complex arithmetic, the
//! arbitrary-length (Bluestein) DFT, the DCT-III as a first-class
//! transform, convolution/correlation, descriptive statistics,
//! interpolation, RANSAC and dynamic time warping.
//!
//! Part of the `pith` zero-dependency hashing kit: every crate in
//! this suite builds without a single registry package, enforced by
//! `scripts/check-zero-deps.py`.
//!
//! The crate is `std`, not `no_std`: the FFT twiddle factors and the
//! DCT kernel need `f64::sin`/`f64::cos`, which `core` does not provide.
//! It still allocates only through `Vec` at API boundaries — all
//! in-place transforms keep working-set memory caller-owned.
//!
//! # The pinned conventions
//!
//! * [`median`]: for even `n` the result is the **lower** middle
//!   element `sorted[n/2 - 1]`, never the mean of the two middles. The
//!   module docs carry the argument; the test on `[1,2,3,4]` pins it.
//! * [`fft`]/[`ifft`]: forward uses `e^(−2πi/N)`, inverse conjugates —
//!   `ifft(fft(x)) == x` exactly, no 1/N on the forward path.
//!   [`fft_arbitrary`]/[`ifft_arbitrary`] lift the power-of-two
//!   restriction to every `n ≥ 1` with the same sign convention.
//! * [`dct2`]/[`idct2`]: **orthonormal** DCT-II/DCT-III, so the inverse
//!   is the transposed kernel and `idct2(dct2(x)) == x`;
//!   [`dct3`]/[`dct3_2d`] are the same DCT-III kernel named as the
//!   first-class transform.
//! * [`convolve`]/[`correlate`]: full-support output, `n + m − 1`;
//!   small inputs run the direct definition, large ones the
//!   arbitrary-length FFT path, agreeing within tolerance.
//! * [`mean`]/[`variance`]/[`covariance`]: sample form (the `n − 1`
//!   denominator); empty/underdetermined inputs return `None`, the
//!   [`median`] convention.
//! * [`ransac_line`]: seeded determinism — the fleet's SplitMix64, a
//!   fixed scan order, ties to the first model found; a seeded run
//!   replays bit-for-bit on every platform.
//! * [`solve3`]/[`inverse3`]: singularity is scale-relative,
//!   `pivot <= ε·s` / `|det| <= ε·s³`, and reported as `None` — a
//!   degenerate RANSAC triplet is data, not an error.

#![deny(unsafe_code)]
#![deny(missing_docs)]

mod bluestein;
mod complex;
mod conv;
mod dct;
mod dct3;
mod dtw;
mod fft;
mod interp;
mod linalg;
mod median;
mod ransac;
mod solve3;
mod stats;

pub mod ffi;
pub mod ffi_jni;

pub use bluestein::{fft_arbitrary, ifft_arbitrary};
pub use complex::Complex;
pub use conv::{convolve, correlate};
pub use dct::{dct2, dct2_2d, dct2_in_place, idct2, idct2_2d};
pub use dct3::{dct3, dct3_2d};
pub use dtw::dtw_distance;
pub use fft::{fft, fft_real, ifft};
pub use interp::{lagrange_eval, lerp};
pub use linalg::{IDENTITY3, det3, inverse3, mat3_mul, mat3_mul_vec, transpose3};
pub use median::{median, median_copy};
pub use ransac::{LineFit, SplitMix64, ransac_line};
pub use solve3::solve3;
pub use stats::{covariance, mean, std_dev, variance};
