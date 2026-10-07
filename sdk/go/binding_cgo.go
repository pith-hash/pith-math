// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash

//go:build !windows && cgo

package pithmath

/*
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef int32_t (*pith_alloc_fn)(const double *, size_t, double **, size_t *);
typedef int32_t (*pith_alloc2d_fn)(const double *, size_t, size_t, size_t, double **, size_t *);
typedef int32_t (*pith_scalar_fn)(const double *, size_t, double *);
typedef void (*pith_free_fn)(double *, size_t);

static int32_t pith_call_alloc(void *fn, const double *in, size_t len,
                               double **out, size_t *out_len) {
    return ((pith_alloc_fn)fn)(in, len, out, out_len);
}

static int32_t pith_call_alloc2d(void *fn, const double *in, size_t len,
                                 size_t w, size_t h, double **out, size_t *out_len) {
    return ((pith_alloc2d_fn)fn)(in, len, w, h, out, out_len);
}

static int32_t pith_call_scalar(void *fn, const double *in, size_t len, double *out) {
    return ((pith_scalar_fn)fn)(in, len, out);
}

static void pith_call_free(void *fn, double *ptr, size_t len) {
    ((pith_free_fn)fn)(ptr, len);
}
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// openCdylib dlopens libPath with error text surfaced verbatim.
func openCdylib(libPath string) (unsafe.Pointer, error) {
	cPath := C.CString(libPath)
	defer C.free(unsafe.Pointer(cPath))
	handle := C.dlopen(cPath, C.RTLD_NOW|C.RTLD_LOCAL)
	if handle == nil {
		msg := "unknown dlopen failure"
		if e := C.dlerror(); e != nil {
			msg = C.GoString(e)
		}
		return nil, fmt.Errorf("pithmath: dlopen(%s): %s", libPath, msg)
	}
	return handle, nil
}

// resolveSymbol dlsyms one name, erroring with the operation and path
// spelled out.
func resolveSymbol(handle unsafe.Pointer, libPath, name string) (unsafe.Pointer, error) {
	cName := C.CString(name)
	sym := C.dlsym(handle, cName)
	C.free(unsafe.Pointer(cName))
	if sym == nil {
		return nil, fmt.Errorf("pithmath: symbol %s missing from %s", name, libPath)
	}
	return sym, nil
}

// ffiAlloc opens the cdylib, resolves op and calls it. The handle is
// released before returning; repeated calls reuse the loader's own
// refcount.
func ffiAlloc(libPath, op string, data *float64, n int, out **byte, outLen *uintptr) (int32, error) {
	handle, err := openCdylib(libPath)
	if err != nil {
		return 0, err
	}
	defer C.dlclose(handle)

	sym, err := resolveSymbol(handle, libPath, op)
	if err != nil {
		return 0, err
	}
	var cOut *C.double
	var cLen C.size_t
	rc := C.pith_call_alloc(sym, (*C.double)(unsafe.Pointer(data)), C.size_t(n), &cOut, &cLen)
	*out = (*byte)(unsafe.Pointer(cOut))
	*outLen = uintptr(cLen)
	return int32(rc), nil
}

// ffiAlloc2d is ffiAlloc with the extra w/h geometry arguments.
func ffiAlloc2d(libPath, op string, data *float64, n int, w, h int, out **byte, outLen *uintptr) (int32, error) {
	handle, err := openCdylib(libPath)
	if err != nil {
		return 0, err
	}
	defer C.dlclose(handle)

	sym, err := resolveSymbol(handle, libPath, op)
	if err != nil {
		return 0, err
	}
	var cOut *C.double
	var cLen C.size_t
	rc := C.pith_call_alloc2d(sym, (*C.double)(unsafe.Pointer(data)), C.size_t(n),
		C.size_t(w), C.size_t(h), &cOut, &cLen)
	*out = (*byte)(unsafe.Pointer(cOut))
	*outLen = uintptr(cLen)
	return int32(rc), nil
}

// ffiScalar resolves op and calls it, writing through a typed f64
// slot.
func ffiScalar(libPath, op string, data *float64, n int, out *float64) (int32, error) {
	handle, err := openCdylib(libPath)
	if err != nil {
		return 0, err
	}
	defer C.dlclose(handle)

	sym, err := resolveSymbol(handle, libPath, op)
	if err != nil {
		return 0, err
	}
	var slot C.double
	rc := C.pith_call_scalar(sym, (*C.double)(unsafe.Pointer(data)), C.size_t(n), &slot)
	*out = float64(slot)
	return int32(rc), nil
}

// ffiFree releases a buffer handed out by ffiAlloc/ffiAlloc2d. Null is
// accepted (the cdylib ignores it), matching the C contract.
func ffiFree(libPath string, ptr *byte, n uintptr) {
	handle, err := openCdylib(libPath)
	if err != nil {
		return
	}
	defer C.dlclose(handle)

	sym, err := resolveSymbol(handle, libPath, "pith_math_free")
	if err != nil {
		return
	}
	C.pith_call_free(sym, (*C.double)(unsafe.Pointer(ptr)), C.size_t(n))
}
