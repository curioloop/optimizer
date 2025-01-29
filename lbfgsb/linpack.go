// Copyright ©2025 curioloop. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lbfgsb

import (
	"math"
)

const (
	solveLowerN = 0b00
	solveUpperN = 0b01
	solveLowerT = 0b10
	solveUpperT = 0b11
)

// dtrsl solves systems of the form
//
//	T * x = b or Tᵀ * x = b
//
// where T is a triangular matrix of order n.
//
// on entry
//
//	t         double precision(n,ldt)
//	          t contains the matrix of the system. the zero
//	          elements of the matrix are not referenced, and
//	          the corresponding elements of the array can be
//	          used to store other information.
//
//	ldt       integer
//	          ldt is the leading dimension of the array t.
//
//	n         integer
//	          n is the order of the system.
//
//	b         double precision(n).
//	          b contains the right hand side of the system.
//
//	job       integer
//	          job specifies what kind of system is to be solved.
//	          if job is
//
//	               00   solve T * x = b, T is lower triangular,
//	               01   solve T * x = b, T is upper triangular,
//	               10   solve Tᵀ * x = b, T is lower triangular,
//	               11   solve Tᵀ * x = b, T is upper triangular.
//
// on return
//
//	b         b contains the solution, if info .eq. 0.
//	          otherwise b is unaltered.
//
//	info      integer
//	          info contains zero if the system is nonsingular.
//	          otherwise info contains the index of
//	          the first zero diagonal element of t.
func dtrsl(t []float64, ldt, n int, b []float64, ldb int, job int) (info int) {

	tn := uint(ldt * n)
	if len(t) <= 0 || len(b) <= 0 || tn > uint(len(t)) {
		panic("bound check error")
	}

	// Check for zero diagonal elements
	for idx := uint(0); idx < tn; idx += uint(1 + ldt) {
		if t[idx] == 0.0 {
			info = 1 + int(idx)/(1+ldt)
			return // Singular matrix detected
		}
	}

	switch job {
	case solveLowerN: // Solve T * x = b for T lower triangular
		b[0] /= t[0]
		for j := 1; j < n; j++ {
			temp := -b[(j-1)*ldb]
			daxpy(n-j, temp, t[j*ldt+(j-1):], ldt, b[j*ldb:], ldb)
			b[j*ldb] /= t[j*ldt+j]
		}
	case solveUpperN: // Solve T * x = b for T upper triangular
		b[(n-1)*ldb] /= t[(n-1)*ldt+(n-1)]
		for j := n - 2; j >= 0; j-- {
			temp := -b[(j+1)*ldb]
			daxpy(j+1, temp, t[j+1:], ldt, b, ldb)
			b[j*ldb] /= t[j*ldt+j]
		}
	case solveLowerT: // Solve trans(T) * x = b for T lower triangular
		b[(n-1)*ldb] /= t[(n-1)*ldt+(n-1)]
		for j := n - 2; j >= 0; j-- {
			temp := ddot((n-1)-j, t[(j+1)*ldt+j:], ldt, b[(j+1)*ldb:], ldb)
			b[j*ldb] = (b[j*ldb] - temp) / t[j*ldt+j]
		}
	case solveUpperT: // Solve trans(T) * x = b for T upper triangular
		b[0] /= t[0]
		for j := 1; j < n; j++ {
			temp := ddot(j, t[j:], ldt, b, ldb)
			b[j*ldb] = (b[j*ldb] - temp) / t[j*ldt+j]
		}
	default:
		info = -1
	}
	return
}

// dpofa factors a double precision symmetric positive definite matrix A = Rᵀ * R.
//
//	on entry
//
//	   a       double precision(n, lda)
//	           the symmetric matrix to be factored.  only the
//	           diagonal and upper triangle are used.
//
//	   lda     integer
//	           the leading dimension of the array  a .
//
//	   n       integer
//	           the order of the matrix  a .
//
//	on return
//
//	   a       an upper triangular matrix  R  so that  A = Rᵀ * R
//	           where  trans(r)  is the transpose.
//	           the strict lower triangle is unaltered.
//	           if  info .ne. 0 , the factorization is not complete.
//
//	   info    integer
//	           = 0  for normal return.
//	           = k  signals an error condition.  the leading minor
//	                of order  k  is not positive definite.
func dpofa(a []float64, lda, n int) (info int) {
	if n > len(a) {
		panic("bound check error")
	}
	for j := 0; j < n; j++ {
		info = j + 1
		s := 0.0
		for k := 0; k < j; k++ {
			t := a[k*lda+j] - ddot(k, a[k:], lda, a[j:], lda)
			t /= a[k*lda+k]
			a[k*lda+j] = t
			s += t * t
		}
		s = a[j*lda+j] - s
		if s <= 0.0 {
			return
		}
		a[j*lda+j] = math.Sqrt(s)
	}
	return 0
}

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
