// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash

// Package pithmath provides Go bindings for the pith-math Rust cdylib:
// FFT, DCT-II, order statistics and 3×3 linear algebra.
//
// The single Rust core (built by `cargo build --release`) is loaded at
// runtime; the package carries zero module dependencies. On unix the
// cdylib is opened with dlopen through cgo, on Windows with
// LoadLibrary through the standard syscall package — both resolve the
// library through the same discovery chain, so `go build ./... &&
// go test ./...` works unchanged on every OS the CD matrix builds.
//
// Discovery order (the suite's cdylib convention):
//
//  1. PITH_CDYLIB — an explicit cdylib file path;
//  2. PITH_CDYLIB_DIR — a directory scanned for the cdylib names (the
//     CD pipeline points this at target/release);
//  3. <repo root>/target/release — the repository working-tree layout,
//     anchored at this package's source directory, so a source
//     checkout runs against a local cargo build unconfigured.
//
// Every transform takes and returns plain []float64 buffers; the
// scalar order statistics (Median, Det3) return a float64. Non-zero
// FFI statuses come back as *FfiError carrying the raw status code —
// the cdylib never panics through this boundary.
package pithmath

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"unsafe"
)

// Status codes returned by the cdylib's C ABI.
const (
	// StatusOK: success.
	StatusOK int32 = 0
	// StatusInvalid: a caller argument is invalid (a null pointer, an
	// empty input, an odd or non-power-of-two transform length, a
	// geometry mismatch).
	StatusInvalid int32 = -1
	// StatusRejected: the core refused the input (a singular matrix).
	StatusRejected int32 = -2
)

// cdylibNames are the file names cargo may drop into the build
// directory, per platform (windows / linux / macOS).
var cdylibNames = []string{"pith_math.dll", "libpith_math.so", "libpith_math.dylib"}

// FfiError reports a non-zero status code from the cdylib.
type FfiError struct {
	// Op is the FFI operation name.
	Op string
	// Status is the raw status code the FFI returned.
	Status int32
}

func (e *FfiError) Error() string {
	kind := "unknown failure"
	switch e.Status {
	case StatusInvalid:
		kind = "invalid argument"
	case StatusRejected:
		kind = "input rejected"
	}
	return fmt.Sprintf("%s failed: %s (status %d)", e.Op, kind, e.Status)
}

// callAlloc runs one flat-array op: hands the input to the cdylib,
// copies the handed-out buffer into a fresh Go slice (element count =
// byte count / 8) and releases the buffer.
func callAlloc(op string, x []float64) ([]float64, error) {
	libPath, err := locate()
	if err != nil {
		return nil, err
	}
	var dataPtr *float64
	if len(x) > 0 {
		dataPtr = &x[0]
	}
	var out *byte
	var outLen uintptr
	status, err := ffiAlloc(libPath, op, dataPtr, len(x), &out, &outLen)
	if err != nil {
		return nil, err
	}
	if status != StatusOK {
		return nil, &FfiError{Op: op, Status: status}
	}
	buf := takeBuffer(libPath, out, outLen)
	return buf, nil
}

// callAlloc2d is callAlloc with the extra w/h geometry arguments.
func callAlloc2d(op string, x []float64, w, h int) ([]float64, error) {
	libPath, err := locate()
	if err != nil {
		return nil, err
	}
	var dataPtr *float64
	if len(x) > 0 {
		dataPtr = &x[0]
	}
	var out *byte
	var outLen uintptr
	status, err := ffiAlloc2d(libPath, op, dataPtr, len(x), w, h, &out, &outLen)
	if err != nil {
		return nil, err
	}
	if status != StatusOK {
		return nil, &FfiError{Op: op, Status: status}
	}
	buf := takeBuffer(libPath, out, outLen)
	return buf, nil
}

// takeBuffer copies the handed-out cdylib buffer into a Go slice and
// releases the buffer through pith_math_free.
func takeBuffer(libPath string, out *byte, outLen uintptr) []float64 {
	count := int(outLen) / 8
	buf := make([]float64, count)
	if count > 0 {
		src := unsafe.Slice((*float64)(unsafe.Pointer(out)), count)
		copy(buf, src)
	}
	ffiFree(libPath, out, outLen)
	return buf
}

// callScalar runs one scalar-out op.
func callScalar(op string, x []float64) (float64, error) {
	libPath, err := locate()
	if err != nil {
		return 0, err
	}
	var dataPtr *float64
	if len(x) > 0 {
		dataPtr = &x[0]
	}
	var slot float64
	status, err := ffiScalar(libPath, op, dataPtr, len(x), &slot)
	if err != nil {
		return 0, err
	}
	if status != StatusOK {
		return 0, &FfiError{Op: op, Status: status}
	}
	return slot, nil
}

// FindCdylib locates the cdylib through the suite's discovery chain.
func FindCdylib() (string, error) {
	if p := os.Getenv("PITH_CDYLIB"); p != "" {
		if st, err := os.Stat(p); err == nil && st.Mode().IsRegular() {
			return filepath.Abs(p)
		}
	}
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("pithmath: cannot locate the package source directory")
	}
	pkgDir := filepath.Dir(thisFile)
	repoRoot := filepath.Dir(filepath.Dir(pkgDir)) // sdk/go -> sdk -> repo root

	var dirs []string
	if env := os.Getenv("PITH_CDYLIB_DIR"); env != "" {
		dirs = append(dirs, env)
		if !filepath.IsAbs(env) {
			dirs = append(dirs, filepath.Join(repoRoot, env))
		}
	}
	dirs = append(dirs, filepath.Join(repoRoot, "target", "release"))
	for _, dir := range dirs {
		for _, name := range cdylibNames {
			p := filepath.Join(dir, name)
			if st, err := os.Stat(p); err == nil && st.Mode().IsRegular() {
				return p, nil
			}
		}
	}
	return "", fmt.Errorf(
		"pithmath: no cdylib found (searched PITH_CDYLIB, PITH_CDYLIB_DIR and <repo>/target/release); run `cargo build --release` first",
	)
}

// locate resolves the cdylib path once per process.
var locate = sync.OnceValues(FindCdylib)

// Dct2 computes the 1D orthonormal DCT-II of x. An empty input is
// StatusInvalid.
func Dct2(x []float64) ([]float64, error) {
	return callAlloc("pith_math_dct2", x)
}

// Idct2 computes the 1D orthonormal DCT-III — the exact inverse of
// Dct2.
func Idct2(x []float64) ([]float64, error) {
	return callAlloc("pith_math_idct2", x)
}

// Dct22D computes the separable 2D orthonormal DCT-II over a w×h
// row-major matrix; len(x) must equal w*h.
func Dct22D(x []float64, w, h int) ([]float64, error) {
	return callAlloc2d("pith_math_dct2_2d", x, w, h)
}

// Fft computes the forward DFT over interleaved complex pairs
// [re, im, …]; the complex count must be a non-zero power of two.
func Fft(x []float64) ([]float64, error) {
	return callAlloc("pith_math_fft", x)
}

// Ifft computes the inverse DFT, the exact (within ulps) inverse of
// Fft.
func Ifft(x []float64) ([]float64, error) {
	return callAlloc("pith_math_ifft", x)
}

// FftReal computes the forward DFT of a real signal and returns the
// full 2·len(x) interleaved spectrum; len(x) must be a non-zero power
// of two.
func FftReal(x []float64) ([]float64, error) {
	return callAlloc("pith_math_fft_real", x)
}

// Solve3 solves the 3×3 system a·x = b (row-major a). A singular
// system is StatusRejected — ordinary RANSAC data, not a caller bug.
func Solve3(a, b []float64) ([]float64, error) {
	flat := make([]float64, 0, len(a)+len(b))
	flat = append(flat, a...)
	flat = append(flat, b...)
	return callAlloc("pith_math_solve3", flat)
}

// Det3 computes the determinant of the row-major 3×3 matrix m.
func Det3(m []float64) (float64, error) {
	return callScalar("pith_math_det3", m)
}

// Inverse3 computes the inverse of the row-major 3×3 matrix m;
// singular input is StatusRejected.
func Inverse3(m []float64) ([]float64, error) {
	return callAlloc("pith_math_inverse3", m)
}

// Transpose3 transposes the row-major 3×3 matrix m.
func Transpose3(m []float64) ([]float64, error) {
	return callAlloc("pith_math_transpose3", m)
}

// Mat3Mul computes the matrix product a·b of two row-major 3×3
// factors.
func Mat3Mul(a, b []float64) ([]float64, error) {
	flat := make([]float64, 0, len(a)+len(b))
	flat = append(flat, a...)
	flat = append(flat, b...)
	return callAlloc("pith_math_mat3_mul", flat)
}

// Mat3MulVec computes the matrix-vector product m·v.
func Mat3MulVec(m, v []float64) ([]float64, error) {
	flat := make([]float64, 0, len(m)+len(v))
	flat = append(flat, m...)
	flat = append(flat, v...)
	return callAlloc("pith_math_mat3_mul_vec", flat)
}

// Median returns the median of x — the lower middle element for even
// n, never the mean of the two middles. An empty input is
// StatusInvalid.
func Median(x []float64) (float64, error) {
	return callScalar("pith_math_median", x)
}
