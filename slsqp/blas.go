// Copyright ©2025 curioloop. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slsqp

import "math"

// daxpy performs constant times a vector plus a vector operation.
func daxpy(n int, da float64, dx []float64, incx int, dy []float64, incy int) {
	if n <= 0 || da == 0.0 {
		return
	}
	if incx == 1 && incy == 1 {
		m := uint(n % 4)
		if m > uint(len(dx)) || m > uint(len(dy)) {
			panic("bound check error")
		}
		for i := uint(0); i < m; i++ {
			dy[i] += da * dx[i]
		}
		if n < 4 {
			return
		}
		for i := m; i < uint(n); i += 4 {
			x := dx[i : i+4 : i+4]
			y := dy[i : i+4 : i+4]
			y[0] += da * x[0]
			y[1] += da * x[1]
			y[2] += da * x[2]
			y[3] += da * x[3]
		}
	} else {
		lx, ly := uint(incx*(n-1)), uint(incy*(n-1))
		if lx >= uint(len(dx)) || ly >= uint(len(dy)) {
			panic("bound check error")
		}
		ix, iy := uint(0), uint(0)
		for ix <= lx && iy <= ly {
			dy[iy] += da * dx[ix]
			ix += uint(incx)
			iy += uint(incy)
		}
	}
}

// ddot computes the dot product of two vectors.
func ddot(n int, dx []float64, incx int, dy []float64, incy int) (dot float64) {
	if n <= 0 {
		return 0.0
	}
	if incx == 1 && incy == 1 {
		m := uint(n % 5)
		if m > uint(len(dx)) || m > uint(len(dy)) {
			panic("bound check error")
		}
		for i := uint(0); i < m; i++ {
			dot += dx[i] * dy[i]
		}
		if n < 5 {
			return dot
		}
		for i := m; i < uint(n); i += 5 {
			x := dx[i : i+5 : i+5]
			y := dy[i : i+5 : i+5]
			dot += x[0]*y[0] + x[1]*y[1] + x[2]*y[2] + x[3]*y[3] + x[4]*y[4]
		}
	} else {
		lx, ly := uint(incx*(n-1)), uint(incy*(n-1))
		if lx >= uint(len(dx)) || ly >= uint(len(dy)) {
			panic("bound check error")
		}
		ix, iy := uint(0), uint(0)
		for ix <= lx && iy <= ly {
			dot += dx[ix] * dy[iy]
			ix += uint(incx)
			iy += uint(incy)
		}
	}
	return dot
}

// dcopy copies a vector, x, to a vector, y.
func dcopy(n int, dx []float64, incx int, dy []float64, incy int) {
	if n <= 0 {
		return
	}
	if incx == 1 && incy == 1 {
		copy(dy[:n], dx[:n])
	} else {
		lx, ly := uint(incx*(n-1)), uint(incy*(n-1))
		if lx >= uint(len(dx)) || ly >= uint(len(dy)) {
			panic("bound check error")
		}
		ix, iy := uint(0), uint(0)
		for ix <= lx && iy <= ly {
			dy[iy] = dx[ix]
			ix += uint(incx)
			iy += uint(incy)
		}
	}
}

// dscal scales a vector by a constant.
func dscal(n int, da float64, dx []float64, incx int) {
	if n <= 0 || incx <= 0 {
		return
	}
	if incx == 1 {
		m := uint(n % 5)
		if m > uint(len(dx)) {
			panic("bound check error")
		}
		for i := uint(0); i < m; i++ {
			dx[i] *= da
		}
		if n < 5 {
			return
		}
		for i := m; i < uint(n); i += 5 {
			d := dx[i : i+5 : i+5]
			d[0] *= da
			d[1] *= da
			d[2] *= da
			d[3] *= da
			d[4] *= da
		}
	} else {
		l := uint(incx * n)
		if l > uint(len(dx)) {
			panic("bound check error")
		}
		for i := uint(0); i < l; i += uint(incx) {
			dx[i] *= da
		}
	}
}

// dnrm2 computes the Euclidean norm of a vector x.
func dnrm2(n int, x []float64, incx int) float64 {
	if n < 1 || incx < 1 {
		return zero
	}

	m := uint(incx * n)
	if m > uint(len(x)) || len(x) <= 0 {
		panic("bound check error")
	}

	if n == 1 {
		return math.Abs(x[0])
	}

	scale := zero
	ssq := one
	for i := uint(0); i < m; i += uint(incx) {
		if absxi := math.Abs(x[i]); absxi > 0 {
			if scale < absxi {
				sxi := scale / absxi
				ssq = 1 + ssq*sxi*sxi
				scale = absxi
			} else {
				sxi := absxi / scale
				ssq += sxi * sxi
			}
		}
	}

	return scale * math.Sqrt(ssq)
}

// dzero fills vector x with zero.
func dzero(dx []float64) {
	n := uint(len(dx))
	m := n % 5
	if m > n {
		panic("bound check error")
	}
	for i := uint(0); i < m; i++ {
		dx[i] = zero
	}
	if n < 5 {
		return
	}
	for i := m; i < n; i += 5 {
		d := dx[i : i+5 : i+5]
		d[0] = zero
		d[1] = zero
		d[2] = zero
		d[3] = zero
		d[4] = zero
	}
}
