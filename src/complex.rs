//! Complex arithmetic over `f64`.
//!
//! The kit keeps its own type rather than taking a dependency on `num`
//! or `rustfft` — the workspace rule is zero external crates. What
//! started as the two-field record the FFT kernel needed grows here
//! into the arithmetic surface the tier-1 kernels share: the four
//! field operations, the exponential/logarithm pair (principal
//! branch), the square root, and the polar forms.
//!
//! Division uses **Smith's algorithm** (scale by the larger component
//! before the quotient), the robust classic: it cannot overflow for
//! finite inputs whose quotient is finite, unlike the naive
//! `(a·c + b·d)/(c² + d²)` form whose denominator overflows for
//! `|c|, |d| ≥ 2^448` long before the quotient does.
//!
//! [`exp`] and [`ln`](Complex::ln) are mutual inverses on the plane
//! cut along the negative real axis; the imaginary part of
//! `ln(z)` is [`arg`]` ∈ (−π, π]`, the principal argument.

use core::ops::Div;

/// A complex number over `f64`, stored as two contiguous scalars.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Complex {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex {
    /// The zero value.
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };

    /// The multiplicative identity.
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };

    /// Constructs `re + i·im`.
    #[must_use]
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// `e^(i·θ)` — the unit-magnitude twiddle factor builder.
    ///
    /// Computed from a fresh `cos`/`sin` pair every call, never by
    /// repeated multiplication: twiddles accumulated by
    /// `w = w * wlen` drift measurably at N = 4096, while a direct
    /// `cis` call keeps the error at a few ulps per factor.
    #[must_use]
    #[inline]
    pub fn cis(theta: f64) -> Self {
        Self {
            re: theta.cos(),
            im: theta.sin(),
        }
    }

    /// `r·e^(i·θ)` — polar construction. `r` may be negative; the
    /// components come straight from the product, so the sign lands
    /// where the math puts it.
    #[must_use]
    #[inline]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    /// Complex conjugate `re − i·im`.
    #[must_use]
    #[inline]
    pub const fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// `|z|²` — cheaper than [`norm`](Self::norm), and the quantity
    /// Parseval's identity and the audio peak picker actually compare.
    #[must_use]
    #[inline]
    pub fn norm_sq(self) -> f64 {
        self.re.mul_add(self.re, self.im * self.im)
    }

    /// `|z|`. Uses [`f64::hypot`] so a huge component does not overflow
    /// the squared intermediate.
    #[must_use]
    #[inline]
    pub fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// The principal argument `arg z ∈ (−π, π]`, with `arg 0 = 0`.
    ///
    /// [`f64::atan2`] already implements exactly that convention, so
    /// this is a direct delegation — the branch cut lands on the
    /// negative real axis, approached from above.
    #[must_use]
    #[inline]
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    /// Multiplies every part by the real scalar `s`.
    #[must_use]
    #[inline]
    pub const fn scale(self, s: f64) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }

    /// `1/z` by Smith's scaling — see the module note.
    ///
    /// `1/0 = NaN + NaN·i`, the IEEE answer carried through.
    #[must_use]
    pub fn recip(self) -> Self {
        // Scale whichever component of the denominator is larger so
        // the sum `d` cannot overflow; `r ≤ 1` in both branches.
        if self.re.abs() >= self.im.abs() {
            let r = self.im / self.re;
            let d = self.re + self.im * r;
            Self {
                re: 1.0 / d,
                im: -r / d,
            }
        } else {
            let r = self.re / self.im;
            let d = self.im + self.re * r;
            Self {
                re: r / d,
                im: -1.0 / d,
            }
        }
    }

    /// `e^z = e^re·(cos im + i·sin im)`.
    ///
    /// One real `exp` plus a fresh `cos`/`sin` pair — the same
    /// no-accumulated-drift reasoning as [`Complex::cis`].
    #[must_use]
    #[inline]
    pub fn exp(self) -> Self {
        let m = self.re.exp();
        Self {
            re: m * self.im.cos(),
            im: m * self.im.sin(),
        }
    }

    /// The principal-branch logarithm `ln|z| + i·arg z`.
    ///
    /// `ln 0 = −inf + 0·i` (sign of the zero imaginary part follows
    /// `arg 0 = 0`); the cut along the negative real axis is inherited
    /// from [`arg`](Self::arg).
    #[must_use]
    #[inline]
    pub fn ln(self) -> Self {
        Self {
            re: self.norm().ln(),
            im: self.arg(),
        }
    }

    /// The principal square root, `re ≥ 0`.
    ///
    /// Half-angle form: with `d = √((|z| + re)/2)` the root is
    /// `d + i·im/(2d)` when `re ≥ 0`, and `|im|/(2d) + i·sign(im)·d`
    /// otherwise — both components stay finite for finite `z`, unlike
    /// the naive `im/(2·√z)` when the root's real part is zero.
    /// `sqrt(-4) = 2i`, `sqrt(0) = 0`.
    #[must_use]
    pub fn sqrt(self) -> Self {
        if self.re == 0.0 && self.im == 0.0 {
            return Self::ZERO;
        }
        if self.re >= 0.0 {
            let d = (0.5 * (self.norm() + self.re)).sqrt();
            Self {
                re: d,
                im: 0.5 * self.im / d,
            }
        } else {
            let d = (0.5 * (self.norm() - self.re)).sqrt();
            // The root's imaginary part carries the sign of the input's,
            // negative zero included: the branch cut approaches the
            // negative real axis from both sides consistently.
            let im = if self.im.is_sign_negative() { -d } else { d };
            Self {
                re: 0.5 * self.im / im,
                im,
            }
        }
    }

    /// Integer power by squaring; negative exponents go through
    /// [`recip`](Self::recip). `z⁰ = 1`, including `0⁰ = 1` (the
    /// combinatorial convention, and the one `powi` on the reals
    /// keeps).
    #[must_use]
    pub fn powi(self, mut n: i32) -> Self {
        if n < 0 {
            return self.recip().powi(-n);
        }
        let mut base = self;
        let mut acc = Self::ONE;
        while n > 0 {
            if n & 1 == 1 {
                acc = acc * base;
            }
            base = base * base;
            n >>= 1;
        }
        acc
    }

    /// Real power via the principal branch: `z^p = e^(p·ln z)`.
    ///
    /// Multi-valued on the cut the way complex analysis says; this is
    /// the same branch [`ln`](Self::ln) publishes.
    #[must_use]
    #[inline]
    pub fn powf(self, p: f64) -> Self {
        self.ln().scale(p).exp()
    }
}

impl std::ops::Add for Complex {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;

    /// `(re + i·im)·(rhs.re + i·rhs.im)`; one `mul_add` per component
    /// keeps the rounding tight.
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re.mul_add(rhs.re, -(self.im * rhs.im)),
            im: self.re.mul_add(rhs.im, self.im * rhs.re),
        }
    }
}

impl Div for Complex {
    type Output = Self;

    /// `self / rhs` by Smith's scaling — the same robust shape as
    /// [`recip`](Complex::recip), generalized to a complex numerator.
    /// `x/0 = NaN + NaN·i` component-wise, the IEEE answer.
    #[inline]
    fn div(self, rhs: Self) -> Self {
        if rhs.re.abs() >= rhs.im.abs() {
            let r = rhs.im / rhs.re;
            let d = rhs.re + rhs.im * r;
            Self {
                re: (self.re + self.im * r) / d,
                im: (self.im - self.re * r) / d,
            }
        } else {
            let r = rhs.re / rhs.im;
            let d = rhs.im + rhs.re * r;
            Self {
                re: (self.re * r + self.im) / d,
                im: (self.im * r - self.re) / d,
            }
        }
    }
}

impl std::ops::Neg for Complex {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}
