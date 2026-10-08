// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash
package pith.math;

import java.io.File;

/**
 * pith-math SDK: FFT, DCT, complex arithmetic, convolution, statistics,
 * interpolation, RANSAC and DTW through the pith-math cdylib over JNI.
 *
 * <p>The single Rust core (built by {@code cargo build --release}) is
 * loaded once at class initialization; this artifact carries zero
 * runtime dependencies. Discovery order mirrors the suite's cdylib
 * convention:
 *
 * <ol>
 *   <li>{@code PITH_CDYLIB} — an explicit cdylib file path;
 *   <li>{@code PITH_CDYLIB_DIR} — a directory scanned for the cdylib
 *       names;
 *   <li>{@code <repo root>/target/release} resolved against the
 *       working directory, so a source checkout runs against a local
 *       cargo build with no configuration.
 * </ol>
 *
 * <p>Every method maps 1:1 onto a {@code Java_pith_math_PithMath_*}
 * export; the refusal conventions are the cdylib's, proven by the
 * fake-env battery in the Rust tree ({@code tests/ffi_jni_env.rs}):
 *
 * <ul>
 *   <li>a caller bug the core pre-validates — geometry mismatches,
 *       non-power-of-two transform lengths, wrong pair counts, empty
 *       scalar inputs — comes back as an <b>empty array</b> for the
 *       allocating kernels, or {@code NaN} for the scalar kernels
 *       ({@code median}, {@code mean}, {@code lagrange},
 *       {@code complexArg}); the Java caller may pre-validate, these
 *       returns are the documented contract either way;
 *   <li>a data refusal — singular {@code solve3}/{@code inverse3},
 *       {@code complexDiv} through a zero denominator,
 *       {@code complexLog} of zero, a RANSAC run that found no model —
 *       surfaces as {@code null} (allocating) exactly where the core
 *       says the data, not the caller, is the problem;
 *   <li>seeded runs ({@code ransacLine}) replay bit-for-bit on every
 *       platform.
 * </ul>
 */
public final class PithMath {

    private static final String[] CDYLIB_NAMES = {
        "pith_math.dll", "libpith_math.so", "libpith_math.dylib",
    };

    static {
        System.load(findCdylib());
    }

    private PithMath() {
    }

    private static String findCdylib() {
        String explicitPath = System.getenv("PITH_CDYLIB");
        if (explicitPath != null && new File(explicitPath).isFile()) {
            return new File(explicitPath).getAbsolutePath();
        }
        String envDir = System.getenv("PITH_CDYLIB_DIR");
        if (envDir != null) {
            for (String name : CDYLIB_NAMES) {
                File candidate = new File(envDir, name);
                if (candidate.isFile()) {
                    return candidate.getAbsolutePath();
                }
            }
        }
        File repoRoot = new File(System.getProperty("user.dir")).getAbsoluteFile()
                .getParentFile().getParentFile();
        File release = new File(repoRoot, "target" + File.separator + "release");
        for (String name : CDYLIB_NAMES) {
            File candidate = new File(release, name);
            if (candidate.isFile()) {
                return candidate.getAbsolutePath();
            }
        }
        throw new UnsatisfiedLinkError(
                "no pith-math cdylib found (searched PITH_CDYLIB, PITH_CDYLIB_DIR"
                + " and <repo>/target/release); run `cargo build --release` first");
    }

    // -- DCT family ---------------------------------------------------------

    /**
     * 1D orthonormal DCT-II; an empty input is an empty result.
     *
     * @param x the samples
     * @return the transform, same length
     */
    public static native double[] dct2(double[] x);

    /** 1D orthonormal DCT-III — the exact inverse of {@link #dct2}. */
    public static native double[] idct2(double[] x);

    /**
     * 2D separable orthonormal DCT-II over a {@code w × h} row-major
     * matrix; a geometry mismatch is an empty result.
     */
    public static native double[] dct22d(double[] data, int w, int h);

    /**
     * 1D orthonormal DCT-III — the first-class forward transform whose
     * inverse is {@link #dct2}.
     */
    public static native double[] dct3(double[] x);

    /**
     * 2D separable orthonormal DCT-III — the exact inverse of
     * {@link #dct22d}; a geometry mismatch is an empty result.
     */
    public static native double[] dct32d(double[] data, int w, int h);

    // -- FFT family -----------------------------------------------------------

    /**
     * Forward DFT over interleaved complex pairs {@code [re, im, …]};
     * the complex count must be a non-zero power of two — anything
     * else is an empty result.
     */
    public static native double[] fft(double[] x);

    /** Inverse DFT, the exact (within ulps) inverse of {@link #fft}. */
    public static native double[] ifft(double[] x);

    /**
     * Forward DFT of a real signal; the full {@code 2·x.length}
     * interleaved spectrum, the length a non-zero power of two.
     */
    public static native double[] fftReal(double[] x);

    /**
     * Forward DFT of <b>any</b> {@code n ≥ 1} (Bluestein below the
     * radix-2 sizes); interleaved pairs in and out.
     */
    public static native double[] fftN(double[] x);

    /** Inverse of {@link #fftN}. */
    public static native double[] ifftN(double[] x);

    // -- 3×3 linear algebra -----------------------------------------------------

    /**
     * Solves the 3×3 system {@code a·x = b} (row-major {@code a}); a
     * singular system returns {@code null} — ordinary RANSAC data, not
     * a caller bug.
     */
    public static native double[] solve3(double[] a, double[] b);

    /** Determinant of the row-major 3×3 matrix {@code m}. */
    public static native double det3(double[] m);

    /**
     * Inverse of the row-major 3×3 matrix {@code m}; singular input
     * returns {@code null}.
     */
    public static native double[] inverse3(double[] m);

    /** Transpose of the row-major 3×3 matrix {@code m}. */
    public static native double[] transpose3(double[] m);

    /** Matrix product {@code a·b} of two row-major 3×3 factors. */
    public static native double[] mat3Mul(double[] a, double[] b);

    /** Matrix-vector product {@code m·v}. */
    public static native double[] mat3MulVec(double[] m, double[] v);

    // -- statistics ---------------------------------------------------------------

    /**
     * The median of {@code x} — the lower middle element for even
     * {@code n}, never the mean of the two middles; empty input is
     * {@code NaN}.
     */
    public static native double median(double[] x);

    /** Arithmetic mean; empty input is {@code NaN}. */
    public static native double mean(double[] x);

    /** Sample variance (the {@code n − 1} denominator). */
    public static native double variance(double[] x);

    /** Sample covariance of two equal-length series. */
    public static native double covariance(double[] a, double[] b);

    // -- convolution / correlation ---------------------------------------------------

    /**
     * Full-support linear convolution — {@code a.length + b.length − 1}
     * samples (direct below the FFT crossover).
     */
    public static native double[] conv(double[] a, double[] b);

    /** Cross-correlation — {@link #conv} with the flipped kernel. */
    public static native double[] corr(double[] a, double[] b);

    // -- interpolation ------------------------------------------------------------------

    /**
     * Lagrange interpolation through {@code (xs[i], ys[i])} at
     * {@code x}; duplicated abscissae come back as {@code NaN}.
     */
    public static native double lagrange(double[] xs, double[] ys, double x);

    // -- RANSAC / DTW ----------------------------------------------------------------------

    /**
     * Seeded RANSAC line fit over the points {@code (xs[i], ys[i])}:
     * {@code iterations} two-point samples under the inlier
     * {@code threshold}, driven by the {@code seed}ed SplitMix64
     * stream. Returns {@code [slope, intercept, inlierCount]}, or
     * {@code null} when no model was found. Seeded runs replay
     * bit-for-bit on every platform.
     */
    public static native double[] ransacLine(double[] xs, double[] ys,
                                             double threshold, int iterations, long seed);

    /**
     * Dynamic time warping distance (absolute-difference local cost,
     * three monotone steps, optimal path, no window constraint).
     */
    public static native double dtw(double[] a, double[] b);

    // -- complex arithmetic --------------------------------------------------------------------

    /** Complex product of two interleaved pairs {@code [re, im]}. */
    public static native double[] complexMul(double[] z, double[] w);

    /**
     * Complex quotient {@code z / w}; a zero denominator returns
     * {@code null} — the domain refusal, like a singular matrix.
     */
    public static native double[] complexDiv(double[] z, double[] w);

    /** {@code e^z} for the interleaved pair {@code z}. */
    public static native double[] complexExp(double[] z);

    /**
     * Principal natural logarithm {@code [ln|z|, arg z]}; {@code z =
     * 0} returns {@code null} (outside the principal-branch domain).
     */
    public static native double[] complexLog(double[] z);

    /** Principal square root of the interleaved pair {@code z}. */
    public static native double[] complexSqrt(double[] z);

    /**
     * Integer power {@code z^n} by exponentiation-by-squaring
     * ({@code n} may be negative: the reciprocal through {@code n}'s
     * absolute value).
     */
    public static native double[] complexPowi(double[] z, int n);

    /** Principal argument of {@code z} in radians, {@code (−π, π]}. */
    public static native double complexArg(double[] z);
}
