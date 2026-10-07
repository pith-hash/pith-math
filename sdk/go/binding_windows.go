// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash

//go:build windows

package pithmath

import (
	"fmt"
	"syscall"
	"unsafe"
)

// openProc loads libPath and resolves name. The library is released
// before returning: on Windows FreeLibrary unmaps the cdylib, so the
// proc must be used (and its buffer copied out) inside the caller.
func openProc(libPath, name string) (proc uintptr, release func(), err error) {
	lib, err := syscall.LoadLibrary(libPath)
	if err != nil {
		return 0, nil, fmt.Errorf("pithmath: LoadLibrary(%s): %w", libPath, err)
	}
	release = func() { syscall.FreeLibrary(lib) }
	proc, err = syscall.GetProcAddress(lib, name)
	if err != nil {
		release()
		return 0, nil, fmt.Errorf("pithmath: symbol %s missing from %s: %w", name, libPath, err)
	}
	return proc, release, nil
}

// ffiAlloc loads the cdylib with LoadLibrary (absolute path, no PATH
// involvement), resolves op and calls it. The returned buffer stays
// alive in the cdylib until ffiFree.
func ffiAlloc(libPath, op string, data *float64, n int, out **byte, outLen *uintptr) (int32, error) {
	proc, release, err := openProc(libPath, op)
	if err != nil {
		return 0, err
	}
	defer release()

	var cOut *byte
	var cLen uintptr
	rc, _, _ := syscall.SyscallN(proc,
		uintptr(unsafe.Pointer(data)),
		uintptr(n),
		uintptr(unsafe.Pointer(&cOut)),
		uintptr(unsafe.Pointer(&cLen)),
	)
	*out = cOut
	*outLen = cLen
	return int32(rc), nil
}

// ffiAlloc2d is ffiAlloc with the extra w/h geometry arguments.
func ffiAlloc2d(libPath, op string, data *float64, n int, w, h int, out **byte, outLen *uintptr) (int32, error) {
	proc, release, err := openProc(libPath, op)
	if err != nil {
		return 0, err
	}
	defer release()

	var cOut *byte
	var cLen uintptr
	rc, _, _ := syscall.SyscallN(proc,
		uintptr(unsafe.Pointer(data)),
		uintptr(n),
		uintptr(w),
		uintptr(h),
		uintptr(unsafe.Pointer(&cOut)),
		uintptr(unsafe.Pointer(&cLen)),
	)
	*out = cOut
	*outLen = cLen
	return int32(rc), nil
}

// ffiScalar resolves op and calls it, writing through a typed f64 slot
// (go vet rejects uintptr→unsafe.Pointer round-trips; the slot address
// is taken fresh inside the call).
func ffiScalar(libPath, op string, data *float64, n int, out *float64) (int32, error) {
	proc, release, err := openProc(libPath, op)
	if err != nil {
		return 0, err
	}
	defer release()

	var slot float64
	rc, _, _ := syscall.SyscallN(proc,
		uintptr(unsafe.Pointer(data)),
		uintptr(n),
		uintptr(unsafe.Pointer(&slot)),
	)
	*out = slot
	return int32(rc), nil
}

// ffiFree resolves pith_math_free and releases a buffer handed out by
// ffiAlloc/ffiAlloc2d. Null is accepted (the cdylib ignores it).
func ffiFree(libPath string, ptr *byte, n uintptr) {
	proc, release, err := openProc(libPath, "pith_math_free")
	if err != nil {
		return // the library vanished mid-flight; nothing to free
	}
	defer release()
	syscall.SyscallN(proc, uintptr(unsafe.Pointer(ptr)), n)
}
