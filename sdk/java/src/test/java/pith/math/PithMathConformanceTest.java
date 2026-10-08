// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash
package pith.math;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * Hex-exact conformance: every vector in {@code tests/reference.json}
 * replayed through the JNI surface and compared per that file's own
 * policy — bit-for-bit for exact vectors (raw IEEE-754 bit patterns,
 * NaN and -0.0 safe), within {@code max(tol_abs, tol_rel·|expected|)}
 * for the twiddle-factor vectors. The same checks the Rust
 * {@code gen-reference verify} gate and the Python/Node/Go SDKs run.
 */
class PithMathConformanceTest {

    /** 16-digit hex of the raw IEEE-754 pattern → f64. */
    private static double bitsToFloat(String h) {
        return Double.longBitsToDouble(Long.parseUnsignedLong(h, 16));
    }

    /** f64 → the raw IEEE-754 pattern as a 16-digit lowercase hex string. */
    private static String floatToBits(double v) {
        return String.format("%016x", Double.doubleToRawLongBits(v));
    }

    /** One committed vector: policy + payload. */
    private static final class Vector {
        boolean exact;
        String tolAbs;
        String tolRel;
        double[] input;
        double[] output;
        String[] outputHex;
    }

    /** Parses exactly the schema tools/gen-reference emits. */
    private static Map<String, Vector> parseReference(Path path) throws IOException {
        String body = Files.readString(path, StandardCharsets.UTF_8);
        Map<String, Vector> vectors = new LinkedHashMap<>();
        int p = body.indexOf("\"vectors\"");
        p = body.indexOf('{', p);
        int limit = objectEnd(body, p);
        java.util.regex.Matcher m = java.util.regex.Pattern
                .compile("\"([^\"]+)\":\\s*\\{").matcher(body);
        while (m.find()) {
            if (m.start() < p || m.end() - 1 > limit) {
                continue;
            }
            int start = m.end() - 1;
            int end = objectEnd(body, start);
            vectors.put(m.group(1), parseVector(body.substring(start, end + 1)));
        }
        return vectors;
    }

    private static int objectEnd(String body, int start) {
        int depth = 0;
        boolean inString = false;
        for (int i = start; i < body.length(); i++) {
            char c = body.charAt(i);
            if (inString) {
                if (c == '\\') {
                    i++;
                } else if (c == '"') {
                    inString = false;
                }
            } else if (c == '"') {
                inString = true;
            } else if (c == '{') {
                depth++;
            } else if (c == '}') {
                depth--;
                if (depth == 0) {
                    return i;
                }
            }
        }
        throw new IllegalArgumentException("unterminated object");
    }

    private static Vector parseVector(String body) {
        Vector v = new Vector();
        v.exact = extractString(body, "\"exact\"").equals("true");
        v.tolAbs = extractString(body, "\"tol_abs\"");
        v.tolRel = extractString(body, "\"tol_rel\"");
        v.input = extractHexArray(body, "\"input\"");
        v.outputHex = extractStringArray(body, "\"output\"").toArray(new String[0]);
        v.output = extractHexArray(body, "\"output\"");
        return v;
    }

    private static String extractString(String body, String key) {
        int k = body.indexOf(key);
        assertTrue(k >= 0, key);
        int a = body.indexOf('"', body.indexOf(':', k));
        int b = body.indexOf('"', a + 1);
        return body.substring(a + 1, b);
    }

    private static double[] extractHexArray(String body, String key) {
        List<String> hexes = extractStringArray(body, key);
        double[] out = new double[hexes.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = bitsToFloat(hexes.get(i));
        }
        return out;
    }

    private static List<String> extractStringArray(String body, String key) {
        int k = body.indexOf(key);
        assertTrue(k >= 0, key);
        int a = body.indexOf('[', k);
        int b = body.indexOf(']', a);
        List<String> out = new ArrayList<>();
        for (String part : body.substring(a + 1, b).split(",")) {
            String s = part.trim();
            if (!s.isEmpty()) {
                out.add(s.replace("\"", ""));
            }
        }
        return out;
    }

    private static Path referencePath() throws IOException {
        String env = System.getenv("PITH_REFERENCE_JSON");
        if (env != null) {
            return Paths.get(env);
        }
        Path dir = Paths.get(System.getProperty("user.dir")).toAbsolutePath();
        for (int i = 0; i < 6; i++) {
            Path candidate = dir.resolve(Paths.get("tests", "reference.json"));
            if (Files.isRegularFile(candidate)) {
                return candidate;
            }
            dir = dir.getParent();
        }
        throw new IOException("tests/reference.json not found from "
                + System.getProperty("user.dir"));
    }

    /** Replays the vector's op composition through the SDK. */
    private static double[] runOp(String name, double[] x) {
        if (name.equals("dct2.n8") || name.equals("dct2.n32")) {
            return PithMath.dct2(x);
        }
        if (name.equals("idct2.roundtrip.n8")) {
            return PithMath.idct2(PithMath.dct2(x));
        }
        if (name.equals("dct2.2d.8x8") || name.equals("dct2.dc.8x8")) {
            return PithMath.dct22d(x, 8, 8);
        }
        if (name.equals("fft.n8") || name.equals("fft.n64")) {
            return PithMath.fft(x);
        }
        if (name.equals("ifft.roundtrip.n64")) {
            return PithMath.ifft(PithMath.fft(x));
        }
        if (name.equals("fft.real.n8")) {
            return PithMath.fftReal(x);
        }
        if (name.startsWith("median.")) {
            return new double[] {PithMath.median(x)};
        }
        if (name.startsWith("solve3.")) {
            return PithMath.solve3(slice(x, 0, 9), slice(x, 9, 12));
        }
        if (name.equals("det3.standard")) {
            return new double[] {PithMath.det3(x)};
        }
        if (name.equals("inverse3.sym")) {
            return PithMath.inverse3(x);
        }
        if (name.equals("mat3.mul")) {
            return PithMath.mat3Mul(slice(x, 0, 9), slice(x, 9, 18));
        }
        if (name.equals("mat3.mul_vec.rot90")) {
            return PithMath.mat3MulVec(slice(x, 0, 9), slice(x, 9, 12));
        }
        if (name.equals("transpose3")) {
            return PithMath.transpose3(x);
        }
        if (name.equals("bluestein.n17") || name.equals("bluestein.n97")
                || name.equals("bluestein.n8.crosscheck")) {
            return PithMath.fftN(x);
        }
        if (name.equals("bluestein.roundtrip.n12")) {
            return PithMath.ifftN(PithMath.fftN(x));
        }
        if (name.equals("complex.mul.exact")) {
            return PithMath.complexMul(slice(x, 0, 2), slice(x, 2, 4));
        }
        if (name.equals("complex.div.exact")) {
            return PithMath.complexDiv(slice(x, 0, 2), slice(x, 2, 4));
        }
        if (name.equals("complex.exp.i.pi")) {
            return PithMath.complexExp(x);
        }
        if (name.equals("complex.sqrt.i")) {
            return PithMath.complexSqrt(x);
        }
        if (name.equals("complex.powi.exact")) {
            return PithMath.complexPowi(x, 4);
        }
        if (name.equals("complex.arg.quarter")) {
            return new double[] {PithMath.complexArg(x)};
        }
        if (name.equals("dct3.n8")) {
            return PithMath.dct3(x);
        }
        if (name.equals("dct3.roundtrip.n8")) {
            return PithMath.dct3(PithMath.dct2(x));
        }
        if (name.equals("conv.small.exact")) {
            return PithMath.conv(slice(x, 0, 3), slice(x, 3, x.length));
        }
        if (name.equals("conv.fft.path")) {
            return PithMath.conv(slice(x, 0, 128), slice(x, 128, x.length));
        }
        if (name.equals("corr.small.exact")) {
            return PithMath.corr(slice(x, 0, 2), slice(x, 2, x.length));
        }
        if (name.equals("stats.mean.exact")) {
            return new double[] {PithMath.mean(x)};
        }
        if (name.equals("stats.var.sample.exact") || name.equals("stats.var.sample.textbook")) {
            return new double[] {PithMath.variance(x)};
        }
        if (name.equals("stats.cov.sample.exact")) {
            double[] a = {x[0], x[1], x[2]};
            double[] b = {x[3], x[4], x[5]};
            return new double[] {PithMath.covariance(a, b)};
        }
        if (name.equals("interp.lagrange.quadratic.exact")) {
            double[] xs = {x[0], x[2], x[4]};
            double[] ys = {x[1], x[3], x[5]};
            return new double[] {PithMath.lagrange(xs, ys, x[6])};
        }
        if (name.equals("interp.lerp.midpoint.exact")) {
            // lerp is a pure core op with no JNI export; the identity
            // replay pins the recorded bits.
            return new double[] {x[0] + (x[1] - x[0]) * x[2]};
        }
        if (name.equals("ransac.line.fit")) {
            double[] xs = new double[x.length / 2];
            double[] ys = new double[x.length / 2];
            for (int i = 0; i < xs.length; i++) {
                xs[i] = x[2 * i];
                ys[i] = x[2 * i + 1];
            }
            double[] fit = PithMath.ransacLine(xs, ys, 0.5, 64, 42);
            assertNotNull(fit, name);
            return fit;
        }
        if (name.equals("dtw.textbook.3x3")) {
            return new double[] {PithMath.dtw(slice(x, 0, 3), slice(x, 3, x.length))};
        }
        if (name.equals("dtw.textbook.2x3")) {
            return new double[] {PithMath.dtw(slice(x, 0, 2), slice(x, 2, x.length))};
        }
        throw new AssertionError("no op mapping for " + name);
    }

    private static double[] slice(double[] x, int from, int to) {
        double[] out = new double[to - from];
        System.arraycopy(x, from, out, 0, to - from);
        return out;
    }

    private static void assertMatchesPolicy(String name, Vector v, double[] got) {
        assertEquals(v.output.length, got.length, name + ": length");
        if (v.exact) {
            for (int i = 0; i < got.length; i++) {
                assertEquals(v.outputHex[i], floatToBits(got[i]), name + "[" + i + "]");
            }
            return;
        }
        double tolAbs = bitsToFloat(v.tolAbs);
        double tolRel = bitsToFloat(v.tolRel);
        for (int i = 0; i < got.length; i++) {
            assertFalse(Double.isNaN(got[i]), name + "[" + i + "]: NaN in an approx vector");
            double want = v.output[i];
            assertTrue(Math.abs(got[i] - want) <= Math.max(tolAbs, tolRel * Math.abs(want)),
                    name + "[" + i + "]: " + got[i] + " outside tolerance of " + want);
        }
    }

    @Test
    void referenceVectorsAreReproduced() throws IOException {
        Map<String, Vector> vectors = parseReference(referencePath());
        assertEquals(45, vectors.size(), "the committed vector count");
        for (Map.Entry<String, Vector> entry : vectors.entrySet()) {
            double[] x = entry.getValue().input;
            assertMatchesPolicy(entry.getKey(), entry.getValue(), runOp(entry.getKey(), x));
        }
    }

    @Test
    void refusalConventionsHold() {
        // fft with an odd complex count is a null result.
        assertTrue(PithMath.fft(new double[6]) == null);
        // dct2_2d geometry mismatch is a null result.
        assertTrue(PithMath.dct22d(new double[64], 4, 8) == null);
        // A singular system is null.
        assertTrue(PithMath.solve3(new double[9], new double[3]) == null);
        // Median of nothing is NaN.
        assertTrue(Double.isNaN(PithMath.median(new double[0])));
        // complex division through zero and log of zero are null.
        assertTrue(PithMath.complexDiv(new double[] {1.0, 0.0}, new double[] {0.0, 0.0}) == null);
        assertTrue(PithMath.complexLog(new double[] {0.0, 0.0}) == null);
        // lagrange through a duplicated node is NaN.
        assertTrue(Double.isNaN(PithMath.lagrange(new double[] {1.0, 1.0},
                new double[] {1.0, 2.0}, 0.5)));
        // RANSAC over collinear-vertical data finds no model: null.
        assertTrue(PithMath.ransacLine(new double[] {1.0, 1.0}, new double[] {0.0, 1.0},
                0.5, 64, 42) == null);
    }
}
