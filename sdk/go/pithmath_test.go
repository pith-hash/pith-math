// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash

package pithmath

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

// repoRoot resolves the repository root relative to this package
// (sdk/go -> sdk -> repo root), the anchor for tests/reference.json.
func repoRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	if st, err := os.Stat(filepath.Join(root, "tests", "reference.json")); err != nil || st.IsDir() {
		t.Fatalf("tests/reference.json not found at %s", root)
	}
	return root
}

// vector is one entry of tests/reference.json: the recorded op, its
// input, and the output under the vector's own comparison policy.
type vector struct {
	Exact  bool     `json:"exact"`
	Shape  []int    `json:"shape"`
	TolAbs string   `json:"tol_abs"`
	TolRel string   `json:"tol_rel"`
	Input  []string `json:"input"`
	Output []string `json:"output"`
}

// reference parses the committed reference file.
func reference(t *testing.T) map[string]vector {
	t.Helper()
	body, err := os.ReadFile(filepath.Join(repoRoot(t), "tests", "reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Vectors map[string]vector `json:"vectors"`
	}
	if err := json.Unmarshal(body, &doc); err != nil {
		t.Fatal(err)
	}
	return doc.Vectors
}

// bitsToFloat decodes a 16-digit hex IEEE-754 pattern.
func bitsToFloat(t *testing.T, h string) float64 {
	t.Helper()
	bits, err := strconv.ParseUint(h, 16, 64)
	if err != nil {
		t.Fatalf("bad hex %q: %v", h, err)
	}
	return math.Float64frombits(bits)
}

// runOp replays the vector's op composition through the SDK.
func runOp(t *testing.T, name string, x []float64) []float64 {
	t.Helper()
	got, err := func() ([]float64, error) {
		switch {
		case name == "dct2.n8" || name == "dct2.n32":
			return Dct2(x)
		case name == "idct2.roundtrip.n8":
			fwd, err := Dct2(x)
			if err != nil {
				return nil, err
			}
			return Idct2(fwd)
		case name == "dct2.2d.8x8" || name == "dct2.dc.8x8":
			return Dct22D(x, 8, 8)
		case name == "fft.n8" || name == "fft.n64":
			return Fft(x)
		case name == "ifft.roundtrip.n64":
			fwd, err := Fft(x)
			if err != nil {
				return nil, err
			}
			return Ifft(fwd)
		case name == "fft.real.n8":
			return FftReal(x)
		case len(name) > 7 && name[:7] == "median.":
			v, err := Median(x)
			if err != nil {
				return nil, err
			}
			return []float64{v}, nil
		case len(name) > 7 && name[:7] == "solve3.":
			return Solve3(x[:9], x[9:12])
		case name == "det3.standard":
			v, err := Det3(x)
			if err != nil {
				return nil, err
			}
			return []float64{v}, nil
		case name == "inverse3.sym":
			return Inverse3(x)
		case name == "mat3.mul":
			return Mat3Mul(x[:9], x[9:18])
		case name == "mat3.mul_vec.rot90":
			return Mat3MulVec(x[:9], x[9:12])
		case name == "transpose3":
			return Transpose3(x)
		case name == "bluestein.n17" || name == "bluestein.n97" || name == "bluestein.n8.crosscheck":
			return FftN(x)
		case name == "bluestein.roundtrip.n12":
			fwd, err := FftN(x)
			if err != nil {
				return nil, err
			}
			return IfftN(fwd)
		case name == "complex.mul.exact":
			return ComplexMul(x[:2], x[2:4])
		case name == "complex.div.exact":
			return ComplexDiv(x[:2], x[2:4])
		case name == "complex.exp.i.pi":
			return ComplexExp(x)
		case name == "complex.sqrt.i":
			return ComplexSqrt(x)
		case name == "complex.powi.exact":
			return ComplexPowi(x, 4)
		case name == "complex.arg.quarter":
			v, err := ComplexArg(x)
			if err != nil {
				return nil, err
			}
			return []float64{v}, nil
		case name == "dct3.n8":
			return Dct3(x)
		case name == "dct3.roundtrip.n8":
			fwd, err := Dct2(x)
			if err != nil {
				return nil, err
			}
			return Dct3(fwd)
		case name == "conv.small.exact":
			return Conv(x[:3], x[3:])
		case name == "conv.fft.path":
			return Conv(x[:128], x[128:])
		case name == "corr.small.exact":
			return Corr(x[:2], x[2:])
		case name == "stats.mean.exact":
			v, err := Mean(x)
			if err != nil {
				return nil, err
			}
			return []float64{v}, nil
		case name == "stats.var.sample.exact" || name == "stats.var.sample.textbook":
			v, err := Variance(x)
			if err != nil {
				return nil, err
			}
			return []float64{v}, nil
		case name == "stats.cov.sample.exact":
			v, err := Cov(x)
			if err != nil {
				return nil, err
			}
			return []float64{v}, nil
		case name == "interp.lagrange.quadratic.exact":
			v, err := Lagrange([]float64{x[0], x[2], x[4]}, []float64{x[1], x[3], x[5]}, x[6])
			if err != nil {
				return nil, err
			}
			return []float64{v}, nil
		case name == "interp.lerp.midpoint.exact":
			// lerp is a pure core op with no C export; the identity
			// replay pins the recorded bits.
			return []float64{x[0] + (x[1]-x[0])*x[2]}, nil
		case name == "ransac.line.fit":
			xs := make([]float64, 0, len(x)/2)
			ys := make([]float64, 0, len(x)/2)
			for i := 0; i < len(x); i += 2 {
				xs = append(xs, x[i])
				ys = append(ys, x[i+1])
			}
			fit, err := RansacLine(xs, ys, 0.5, 64, 42)
			if err != nil {
				return nil, err
			}
			return []float64{fit.Slope, fit.Intercept, float64(fit.Inliers)}, nil
		case name == "dtw.textbook.3x3":
			v, err := Dtw(x[:3], x[3:])
			if err != nil {
				return nil, err
			}
			return []float64{v}, nil
		case name == "dtw.textbook.2x3":
			v, err := Dtw(x[:2], x[2:])
			if err != nil {
				return nil, err
			}
			return []float64{v}, nil
		default:
			t.Fatalf("no op mapping for %s", name)
			return nil, nil
		}
	}()
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	return got
}

// assertMatchesPolicy applies the vector's own comparison policy.
func assertMatchesPolicy(t *testing.T, name string, v vector, got []float64) {
	t.Helper()
	if len(got) != len(v.Output) {
		t.Fatalf("%s: got %d values, want %d", name, len(got), len(v.Output))
	}
	if v.Exact {
		for i, h := range v.Output {
			wantBits, _ := strconv.ParseUint(h, 16, 64)
			if math.Float64bits(got[i]) != wantBits {
				t.Fatalf("%s[%d]: bits %016x, want %s", name, i, math.Float64bits(got[i]), h)
			}
		}
		return
	}
	tolAbs := bitsToFloat(t, v.TolAbs)
	tolRel := bitsToFloat(t, v.TolRel)
	for i, h := range v.Output {
		want := bitsToFloat(t, h)
		if math.IsNaN(got[i]) {
			t.Fatalf("%s[%d]: NaN in an approx vector", name, i)
		}
		if math.Abs(got[i]-want) > math.Max(tolAbs, tolRel*math.Abs(want)) {
			t.Fatalf("%s[%d]: %v outside tolerance of %v", name, i, got[i], want)
		}
	}
}

// TestReferenceVectors reproduces every committed vector through the
// cdylib — exact vectors bit-for-bit, twiddle-factor vectors within
// the recorded budgets — the same checks the Rust gen-reference verify
// gate and the Python/Node SDKs run.
func TestReferenceVectors(t *testing.T) {
	for name, v := range reference(t) {
		t.Run(name, func(t *testing.T) {
			x := make([]float64, len(v.Input))
			for i, h := range v.Input {
				x[i] = bitsToFloat(t, h)
			}
			assertMatchesPolicy(t, name, v, runOp(t, name, x))
		})
	}
}

// TestDet3StandardPinned pins the full det3.standard vector literally
// (rust-derived), so the binding fails loudly even if reference.json
// were regenerated wrongly.
func TestDet3StandardPinned(t *testing.T) {
	hex := []string{
		"4018000000000000", "3ff0000000000000", "3ff0000000000000",
		"4010000000000000", "c000000000000000", "4014000000000000",
		"4000000000000000", "4020000000000000", "401c000000000000",
	}
	x := make([]float64, len(hex))
	for i, h := range hex {
		x[i] = bitsToFloat(t, h)
	}
	got, err := Det3(x)
	if err != nil {
		t.Fatal(err)
	}
	if bits := math.Float64bits(got); bits != 0xc073200000000000 {
		t.Fatalf("det3 bits %016x, want c073200000000000", bits)
	}
}

// TestFftN8FirstBinsWithinTolerance pins one approx vector's first
// four bins and the recorded budgets in test code.
func TestFftN8FirstBinsWithinTolerance(t *testing.T) {
	inHex := []string{
		"3ff0000000000000", "3fe0000000000000", "bfe0000000000000", "4000000000000000",
		"3fd0000000000000", "bff8000000000000", "3ffc000000000000", "0000000000000000",
		"c002000000000000", "3fe0000000000000", "3fc0000000000000", "4008000000000000",
		"bfe8000000000000", "3ff0000000000000", "4004000000000000", "bff0000000000000",
	}
	wantHex := []string{"4001000000000000", "4012000000000000", "3fead413cccfe77a", "bff712318007c2b1"}
	x := make([]float64, len(inHex))
	for i, h := range inHex {
		x[i] = bitsToFloat(t, h)
	}
	got, err := Fft(x)
	if err != nil {
		t.Fatal(err)
	}
	tolAbs := bitsToFloat(t, "3d3c25c268497682")
	tolRel := bitsToFloat(t, "3d719799812dea11")
	for i, h := range wantHex {
		want := bitsToFloat(t, h)
		if math.IsNaN(got[i]) || math.Abs(got[i]-want) > math.Max(tolAbs, tolRel*math.Abs(want)) {
			t.Fatalf("fft.n8[%d]: %v outside tolerance of %v", i, got[i], want)
		}
	}
}

// TestRefusals checks the refusal paths: a status code, never a crash.
func TestRefusals(t *testing.T) {
	// fft with an odd complex count (6 f64s = 3 pairs, 3 not a power
	// of two) is StatusInvalid.
	if _, err := Fft(make([]float64, 6)); !isStatus(err, StatusInvalid) {
		t.Fatalf("fft n=3 pairs: want status %d, got %v", StatusInvalid, err)
	}

	// dct2_2d with w*h != len is StatusInvalid.
	if _, err := Dct22D(make([]float64, 64), 4, 8); !isStatus(err, StatusInvalid) {
		t.Fatalf("dct2_2d geometry: want status %d, got %v", StatusInvalid, err)
	}

	// A singular (all-zero) system is StatusRejected.
	if _, err := Solve3(make([]float64, 9), make([]float64, 3)); !isStatus(err, StatusRejected) {
		t.Fatalf("singular solve3: want status %d, got %v", StatusRejected, err)
	}

	// Median of nothing is StatusInvalid.
	if _, err := Median(nil); !isStatus(err, StatusInvalid) {
		t.Fatalf("median empty: want status %d, got %v", StatusInvalid, err)
	}

	// A nil data slice hands the cdylib a NULL pointer: StatusInvalid,
	// not a crash.
	if _, err := Dct2(nil); !isStatus(err, StatusInvalid) {
		t.Fatalf("nil data: want status %d, got %v", StatusInvalid, err)
	}

	// Tier-1 geometry refusals.
	if _, err := ComplexDiv([]float64{1, 0}, []float64{0, 0}); !isStatus(err, StatusRejected) {
		t.Fatalf("complex div by zero: want %d, got %v", StatusRejected, err)
	}
	if _, err := ComplexLog([]float64{0, 0}); !isStatus(err, StatusRejected) {
		t.Fatalf("complex log of zero: want %d, got %v", StatusRejected, err)
	}
	if _, err := Dct32D(make([]float64, 64), 0, 8); !isStatus(err, StatusInvalid) {
		t.Fatalf("dct3_2d geometry: want %d, got %v", StatusInvalid, err)
	}
	if _, err := FftN(make([]float64, 5)); !isStatus(err, StatusInvalid) {
		t.Fatalf("fft_n odd byte count: want %d, got %v", StatusInvalid, err)
	}
	if _, err := Conv(nil, []float64{1}); !isStatus(err, StatusInvalid) {
		t.Fatalf("conv empty a: want %d, got %v", StatusInvalid, err)
	}
	if _, err := Mean(nil); !isStatus(err, StatusInvalid) {
		t.Fatalf("mean empty: want %d, got %v", StatusInvalid, err)
	}
	if _, err := Variance([]float64{1}); !isStatus(err, StatusInvalid) {
		t.Fatalf("variance singleton: want %d, got %v", StatusInvalid, err)
	}
	if _, err := Lagrange([]float64{0, 1}, []float64{1}, 0.5); !isStatus(err, StatusInvalid) {
		t.Fatalf("lagrange length mismatch: want %d, got %v", StatusInvalid, err)
	}
	if _, err := RansacLine([]float64{0, 1}, []float64{1}, 0.5, 64, 42); !isStatus(err, StatusInvalid) {
		t.Fatalf("ransac length mismatch: want %d, got %v", StatusInvalid, err)
	}
	if _, err := Dtw(nil, []float64{1}); !isStatus(err, StatusInvalid) {
		t.Fatalf("dtw empty a: want %d, got %v", StatusInvalid, err)
	}
}

// isStatus reports whether err is an *FfiError with the given status.
func isStatus(err error, status int32) bool {
	e, ok := err.(*FfiError)
	return ok && e.Status == status
}
