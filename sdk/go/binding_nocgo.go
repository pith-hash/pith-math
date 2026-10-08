// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash

//go:build !windows && !cgo

package pithmath

import "fmt"

// ffiAlloc is unavailable without cgo on unix: there is no pure-Go
// dlopen in the standard library. Build with CGO_ENABLED=1 (the CD
// pipeline always does).
func ffiAlloc(string, *byte, int, **byte, *uintptr) (int32, error) {
	return 0, fmt.Errorf("pithmath: cgo is required to load the cdylib on this platform (build with CGO_ENABLED=1)")
}

// ffiAlloc2d mirrors the unavailable alloc.
func ffiAlloc2d(string, *byte, int, int, int, **byte, *uintptr) (int32, error) {
	return 0, fmt.Errorf("pithmath: cgo is required to load the cdylib on this platform (build with CGO_ENABLED=1)")
}

// ffiScalar mirrors the unavailable alloc.
func ffiScalar(string, *byte, int, *float64) (int32, error) {
	return 0, fmt.Errorf("pithmath: cgo is required to load the cdylib on this platform (build with CGO_ENABLED=1)")
}

// ffiAllocLen mirrors the unavailable alloc.
func ffiAllocLen(string, *float64, int, int, **byte, *uintptr) (int32, error) {
	return 0, fmt.Errorf("pithmath: cgo is required to load the cdylib on this platform (build with CGO_ENABLED=1)")
}

// ffiScalarLen mirrors the unavailable alloc.
func ffiScalarLen(string, *float64, int, int, *float64) (int32, error) {
	return 0, fmt.Errorf("pithmath: cgo is required to load the cdylib on this platform (build with CGO_ENABLED=1)")
}

// ffiScalarArg mirrors the unavailable alloc.
func ffiScalarArg(string, *float64, int, float64, *float64) (int32, error) {
	return 0, fmt.Errorf("pithmath: cgo is required to load the cdylib on this platform (build with CGO_ENABLED=1)")
}

// ffiRansac mirrors the unavailable alloc.
func ffiRansac(string, *float64, int, float64, int, uint64, **byte, *uintptr) (int32, error) {
	return 0, fmt.Errorf("pithmath: cgo is required to load the cdylib on this platform (build with CGO_ENABLED=1)")
}

// ffiFree mirrors the unavailable alloc.
func ffiFree(string, *byte, uintptr) {}
