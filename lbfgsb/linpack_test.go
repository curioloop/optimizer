// Copyright ©2025 curioloop. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lbfgsb

import (
	"math"
	"reflect"
	"slices"
	"testing"
)

func TestScal(t *testing.T) {

	x := []float64{
		1, 1,
		1, 1,
		1, 1,
		1, 1,
		1, 1,
		1, 1}

	y := []float64{
		2, 1,
		2, 1,
		2, 1,
		2, 1,
		2, 1,
		2, 1}

	dscal(6, 2, x, 2)
	if !almostEqual(x, y, 1e-10) {
		t.Fatal("scal test fail")
	}

}

func TestAxpy(t *testing.T) {

	{
		x := []float64{1, 2, 3, 4, 5, 6}
		y := []float64{0, 0, 0, 0, 0, 0}
		daxpy(6, 1, x, 1, y, 1)
		if !almostEqual(x, y, 1e-10) {
			t.Fatal("axpy test fail")
		}
	}

	{
		x := []float64{
			1, 1,
			2, 1,
			3, 1,
			4, 1,
			5, 1,
			6, 1}

		y := []float64{
			0, 1,
			0, 1,
			0, 1,
			0, 1,
			0, 1,
			0, 1}

		daxpy(6, 1, x, 2, y, 2)
		if !almostEqual(x, y, 1e-10) {
			t.Fatal("axpy test fail")
		}
	}

}

func TestDot(t *testing.T) {
	{
		x := []float64{1, 2, 3, 4, 5, 6}
		a := ddot(6, x, 1, x, 1)
		b := 91.0
		if a != b {
			t.Fatal("ddot test fail")
		}
	}

	{
		x := []float64{
			1, 1,
			2, 1,
			3, 1,
			4, 1,
			5, 1,
			6, 1}

		a := ddot(6, x, 2, x, 2)
		b := 91.0
		if a != b {
			t.Fatal("ddot test fail")
		}
	}

}

func TestDtrsl(t *testing.T) {

	{
		b := []float64{6, 14, 13, 0}
		T := []float64{
			2, 0, 0, 0,
			3, 4, 0, 0,
			1, 2, 3, 0,
			0, 0, 0, 0}

		x := slices.Repeat(b, 1)
		dtrsl(T, 4, 3, x, 1, solveLowerN)
		dtrmv('L', 'N', 'N', 3, T, 4, x, 1)

		if !almostEqual(x, b, 1e-10) {
			t.Fatal("dtrsl test fail")
		}

		x = slices.Repeat(b, 1)
		dtrsl(T, 4, 3, x, 1, solveLowerT)
		dtrmv('L', 'T', 'N', 3, T, 4, x, 1)
		if !almostEqual(x, b, 1e-10) {
			t.Fatal("dtrsl test fail")
		}
	}

	{
		b := []float64{
			6, 0,
			14, 0,
			13, 0,
			0, 0}
		T := []float64{
			2, 0, 0, 0,
			3, 4, 0, 0,
			1, 2, 3, 0,
			0, 0, 0, 0}

		x := slices.Repeat(b, 1)
		dtrsl(T, 4, 3, x, 2, solveLowerN)
		dtrmv('L', 'N', 'N', 3, T, 4, x, 2)
		if !almostEqual(x, b, 1e-10) {
			t.Fatal("dtrsl test fail")
		}

		x = slices.Repeat(b, 1)
		dtrsl(T, 4, 3, x, 2, solveLowerT)
		dtrmv('L', 'T', 'N', 3, T, 4, x, 2)
		if !almostEqual(x, b, 1e-10) {
			t.Fatal("dtrsl test fail")
		}
	}

	{
		b := []float64{14, 26, 18, 0}
		T := []float64{
			1, 2, 3, 0,
			0, 4, 5, 0,
			0, 0, 6, 0,
			0, 0, 0, 0}

		x := slices.Repeat(b, 1)
		dtrsl(T, 4, 3, x, 1, solveUpperN)
		dtrmv('U', 'N', 'N', 3, T, 4, x, 1)
		if !almostEqual(x, b, 1e-10) {
			t.Fatal("dtrsl test fail")
		}

		x = slices.Repeat(b, 1)
		dtrsl(T, 4, 3, x, 1, solveUpperT)
		dtrmv('U', 'T', 'N', 3, T, 4, x, 1)
		if !almostEqual(x, b, 1e-10) {
			t.Fatal("dtrsl test fail")
		}
	}

	{
		b := []float64{
			14, 0,
			26, 0,
			18, 0,
			0, 0}
		T := []float64{
			1, 2, 3, 0,
			0, 4, 5, 0,
			0, 0, 6, 0,
			0, 0, 0, 0}

		x := slices.Repeat(b, 1)
		dtrsl(T, 4, 3, x, 2, solveUpperN)
		dtrmv('U', 'N', 'N', 3, T, 4, x, 2)
		if !almostEqual(x, b, 1e-10) {
			t.Fatal("dtrsl test fail")
		}

		x = slices.Repeat(b, 1)
		dtrsl(T, 4, 3, x, 2, solveUpperT)
		dtrmv('U', 'T', 'N', 3, T, 4, x, 2)
		if !almostEqual(x, b, 1e-10) {
			t.Fatal("dtrsl test fail")
		}
	}

}

func TestDpofa(t *testing.T) {

	a := []float64{
		4, 2, 1, 0,
		2, 3, 1, 0,
		1, 1, 2, 0,
		0, 0, 0, 0,
	}

	r := slices.Repeat(a, 1)
	dpofa(r, 4, 3)

	d := make([]float64, 4*4)
	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ {
			d[i*4+j] = r[i*4+j]
		}
	}

	dtrmm('L', 'U', 'T', 'N', 3, 3, 1, r, 4, d, 4)
	if !almostEqual(a, d, 1e-15) {
		t.Fatal("dpofa test fail")
	}

}

// dtrmm performs matrix-matrix operation:
// B = alpha * A * B (or B = alpha * B * A based on the side)
func dtrmm(s, ul, tA, d rune, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) {

	switch {
	case s != 'L' && s != 'R':
		panic("bad side")
	case ul != 'U' && ul != 'L':
		panic("bad ul")
	case tA != 'N' && tA != 'T':
		panic("bad trans")
	case d != 'N' && d != 'U':
		panic("bad diag")
	}

	A := func(i, j int) float64 {
		if tA == 'N' {
			return a[i*lda+j]
		} else {
			return a[j*lda+i]
		}
	}

	tmpB := slices.Repeat(b, 1)
	B := func(i, j int) float64 {
		return tmpB[i*ldb+j]
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float64
			switch {
			case s == 'L' && ul == 'U' && tA == 'T':
				fallthrough
			case s == 'L' && ul == 'L' && tA == 'N':
				// b = alpha * A * b
				for k := 0; k <= i; k++ {
					if d == 'U' && i == k {
						sum += alpha * B(k, j)
					} else {
						sum += alpha * A(i, k) * B(k, j)
					}
				}
			case s == 'L' && ul == 'L' && tA == 'T':
				fallthrough
			case s == 'L' && ul == 'U' && tA == 'N':
				// b = alpha * A * b
				for k := i; k < m; k++ {
					if d == 'U' && i == k {
						sum += alpha * B(k, j)
					} else {
						sum += alpha * A(i, k) * B(k, j)
					}
				}
			case s == 'R' && ul == 'U' && tA == 'N':
				fallthrough
			case s == 'R' && ul == 'L' && tA == 'T':
				// b = alpha * b * A
				for k := 0; k <= j; k++ {
					if d == 'U' && j == k {
						sum += alpha * B(i, k)
					} else {
						sum += alpha * B(i, k) * A(k, j)
					}
				}
			case s == 'R' && ul == 'U' && tA == 'T':
				fallthrough
			case s == 'R' && ul == 'L' && tA == 'N':
				for k := j; k < n; k++ {
					if d == 'U' && j == k {
						sum += alpha * B(i, k)
					} else {
						sum += alpha * B(i, k) * A(k, j)
					}
				}
			}
			b[i*ldb+j] = sum
		}
	}
}

func TestTrmm(t *testing.T) {
	tests := []struct {
		name         string
		s, ul, tA, d rune
		alpha        float64
		a            []float64
		lda          int
		b            []float64
		ldb          int
		expected     []float64
	}{
		// Left side tests with 2x3 b matrix
		{"Left Upper NoTrans NonUnit", 'L', 'U', 'N', 'N', 1.0,
			[]float64{1, 2, 3, 0, 4, 5, 0, 0, 6}, 3,
			[]float64{1, 2, 3, 4, 5, 6}, 3,
			[]float64{9, 12, 15, 16, 20, 24}},
		{"Left Upper NoTrans Unit", 'L', 'U', 'N', 'U', 1.0,
			[]float64{1, 2, 3, 0, 1, 5, 0, 0, 1}, 3,
			[]float64{1, 2, 3, 4, 5, 6}, 3,
			[]float64{9, 12, 15, 4, 5, 6}},
		{"Left Lower NoTrans NonUnit", 'L', 'L', 'N', 'N', 1.0,
			[]float64{1, 0, 0, 2, 3, 0, 4, 5, 6}, 3,
			[]float64{1, 2, 3, 4, 5, 6}, 3,
			[]float64{1, 2, 3, 14, 19, 24}},
		{"Left Lower NoTrans Unit", 'L', 'L', 'N', 'U', 1.0,
			[]float64{1, 0, 0, 2, 1, 0, 4, 5, 1}, 3,
			[]float64{1, 2, 3, 4, 5, 6}, 3,
			[]float64{1, 2, 3, 6, 9, 12}},
		{"Left Upper Trans NonUnit", 'L', 'U', 'T', 'N', 1.0,
			[]float64{1, 2, 3, 0, 4, 5, 0, 0, 6}, 3,
			[]float64{1, 2, 3, 4, 5, 6}, 3,
			[]float64{1, 2, 3, 18, 24, 30}},
		{"Left Upper Trans Unit", 'L', 'U', 'T', 'U', 1.0,
			[]float64{1, 2, 3, 0, 1, 5, 0, 0, 1}, 3,
			[]float64{1, 2, 3, 4, 5, 6}, 3,
			[]float64{1, 2, 3, 6, 9, 12}},
		{"Left Lower Trans NonUnit", 'L', 'L', 'T', 'N', 1.0,
			[]float64{1, 0, 0, 2, 3, 0, 4, 5, 6}, 3,
			[]float64{1, 2, 3, 4, 5, 6}, 3,
			[]float64{9, 12, 15, 12, 15, 18}},
		{"Left Lower Trans Unit", 'L', 'L', 'T', 'U', 1.0,
			[]float64{1, 0, 0, 2, 1, 0, 4, 5, 1}, 3,
			[]float64{1, 2, 3, 4, 5, 6}, 3,
			[]float64{9, 12, 15, 4, 5, 6}},
		{"Right Upper NoTrans NonUnit", 'R', 'U', 'N', 'N', 1.0,
			[]float64{1, 2, 3, 0, 4, 5, 0, 0, 6}, 3,
			[]float64{1, 2, 3, 4, 5, 6}, 3,
			[]float64{1, 10, 31, 4, 28, 73}},
		{"Right Upper NoTrans Unit", 'R', 'U', 'N', 'U', 1.0,
			[]float64{1, 2, 3, 0, 1, 5, 0, 0, 1}, 3,
			[]float64{1, 2, 3, 4, 5, 6}, 3,
			[]float64{1, 4, 16, 4, 13, 43}},
		{"Right Lower NoTrans NonUnit", 'R', 'L', 'N', 'N', 1.0,
			[]float64{1, 0, 0, 2, 3, 0, 4, 5, 6}, 3,
			[]float64{1, 2, 3, 4, 5, 6}, 3,
			[]float64{17, 21, 18, 38, 45, 36}},
		{"Right Lower NoTrans Unit", 'R', 'L', 'N', 'U', 1.0,
			[]float64{1, 0, 0, 2, 1, 0, 4, 5, 1}, 3,
			[]float64{1, 2, 3, 4, 5, 6}, 3,
			[]float64{17, 17, 3, 38, 35, 6}},
		{"Right Upper Trans NonUnit", 'R', 'U', 'T', 'N', 1.0,
			[]float64{1, 2, 3, 0, 4, 5, 0, 0, 6}, 3,
			[]float64{1, 2, 3, 4, 5, 6}, 3,
			[]float64{14, 23, 18, 32, 50, 36}},
		{"Right Upper Trans Unit", 'R', 'U', 'T', 'U', 1.0,
			[]float64{1, 2, 3, 0, 1, 5, 0, 0, 1}, 3,
			[]float64{1, 2, 3, 4, 5, 6}, 3,
			[]float64{14, 17, 3, 32, 35, 6}},
		{"Right Lower Trans NonUnit", 'R', 'L', 'T', 'N', 1.0,
			[]float64{1, 0, 0, 2, 3, 0, 4, 5, 6}, 3,
			[]float64{1, 2, 3, 4, 5, 6}, 3,
			[]float64{1, 8, 32, 4, 23, 77}},
		{"Right Lower Trans Unit", 'R', 'L', 'T', 'U', 1.0,
			[]float64{1, 0, 0, 2, 1, 0, 4, 5, 1}, 3,
			[]float64{1, 2, 3, 4, 5, 6}, 3,
			[]float64{1, 4, 17, 4, 13, 47}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dtrmm(tt.s, tt.ul, tt.tA, tt.d, 2, 3, tt.alpha, tt.a, tt.lda, tt.b, tt.ldb)
			for i := range tt.b {
				if tt.b[i] != tt.expected[i] {
					t.Errorf("Test failed! Expected: %v, Got: %v", tt.expected, tt.b)
					break
				}
			}
		})
	}
}

// dtrmv performs a matrix-vector multiplication
// x = A * x (or x = A^T * x), where A is a triangular matrix.
func dtrmv(ul, tA, d rune, n int, a []float64, lda int, x []float64, incX int) {

	switch {
	case ul != 'U' && ul != 'L':
		panic("bad ul")
	case tA != 'N' && tA != 'T':
		panic("bad trans")
	case d != 'N' && d != 'U':
		panic("bad diag")
	}

	A := func(i, j int) float64 {
		if tA == 'N' {
			return a[i*lda+j]
		} else {
			return a[j*lda+i]
		}
	}

	tmpX := slices.Repeat(x, 1)
	X := func(i int) float64 {
		return tmpX[i*incX]
	}
	for i := 0; i < n; i++ {
		sum := X(i)
		if d == 'N' {
			sum *= A(i, i)
		}
		switch {
		case ul == 'U' && tA == 'N':
			fallthrough
		case ul == 'L' && tA == 'T':
			for j := i + 1; j < n; j++ {
				sum += A(i, j) * X(j)
			}
		case ul == 'L' && tA == 'N':
			fallthrough
		case ul == 'U' && tA == 'T':
			for j := 0; j < i; j++ {
				sum += A(i, j) * X(j)
			}
		}
		x[i*incX] = sum
	}
}

func TestDtrmv(t *testing.T) {
	tests := []struct {
		name      string
		ul, tA, d rune
		a         []float64
		lda       int
		x         []float64
		expected  []float64
	}{
		{
			name: "Upper NoTrans NonUnit",
			ul:   'U',
			tA:   'N',
			d:    'N',
			a: []float64{
				1, 2, 3,
				0, 4, 5,
				0, 0, 6,
			},
			lda:      3,
			x:        []float64{1, 2, 3},
			expected: []float64{14, 23, 18},
		},
		{
			name: "Lower NoTrans NonUnit",
			ul:   'L',
			tA:   'N',
			d:    'N',
			a: []float64{
				1, 0, 0,
				2, 3, 0,
				4, 5, 6,
			},
			lda:      3,
			x:        []float64{1, 2, 3},
			expected: []float64{1, 8, 32},
		},
		{
			name: "Upper Trans NonUnit",
			ul:   'U',
			tA:   'T',
			d:    'N',
			a: []float64{
				1, 2, 3,
				0, 4, 5,
				0, 0, 6,
			},
			lda:      3,
			x:        []float64{1, 2, 3},
			expected: []float64{1, 10, 31},
		},
		{
			name: "Lower Trans NonUnit",
			ul:   'L',
			tA:   'T',
			d:    'N',
			a: []float64{
				1, 0, 0,
				2, 3, 0,
				4, 5, 6,
			},
			lda:      3,
			x:        []float64{1, 2, 3},
			expected: []float64{17, 21, 18},
		},
		{
			name: "Upper NoTrans Unit",
			ul:   'U',
			tA:   'N',
			d:    'U',
			a: []float64{
				5, 2, 3,
				0, 5, 5,
				0, 0, 5,
			},
			lda:      3,
			x:        []float64{1, 2, 3},
			expected: []float64{14, 17, 3},
		},
		{
			name: "Upper Trans Unit",
			ul:   'U',
			tA:   'T',
			d:    'U',
			a: []float64{
				5, 2, 3,
				0, 5, 5,
				0, 0, 5,
			},
			lda:      3,
			x:        []float64{1, 2, 3},
			expected: []float64{1, 4, 16},
		},
		{
			name: "Lower Trans Unit",
			ul:   'L',
			tA:   'T',
			d:    'U',
			a: []float64{
				1, 0, 0,
				2, 3, 0,
				4, 5, 6,
			},
			lda:      3,
			x:        []float64{1, 2, 3},
			expected: []float64{17, 17, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dtrmv(tt.ul, tt.tA, tt.d, 3, tt.a, tt.lda, tt.x, 1)
			for i := range tt.x {
				if tt.x[i] != tt.expected[i] {
					t.Errorf("Test failed! Expected: %v, Got: %v", tt.expected, tt.x)
					break
				}
			}
		})
	}
}

func TestDgemv(t *testing.T) {
	A := []float64{
		1, 2, 3,
		4, 5, 6,
	}

	x := []float64{1, 1, 1}
	y := []float64{0, 0}
	dgemv(2, 3, 1.0, A, false, x, 0.0, y)
	if e := []float64{6, 15}; !reflect.DeepEqual(y, e) {
		t.Fatal("GEMV Test failed! Expected:", e, "Got:", y)
	}

	x = []float64{1, 1}
	y = []float64{0, 0, 0}
	dgemv(2, 3, 1.0, A, true, x, 0.0, y)
	if e := []float64{5, 7, 9}; !reflect.DeepEqual(y, e) {
		t.Fatal("GEMV Test failed! Expected:", e, "Got:", y)
	}
}

func dgemv(m, n int, alpha float64, A []float64, trans bool, x []float64, beta float64, y []float64) {
	if len(A) != m*n || trans && len(x) != m || !trans && len(x) != n || trans && len(y) != n || !trans && len(y) != m {
		panic("bad shape")
	}

	get := func(i, j int) float64 {
		return A[i*n+j]
	}

	// 计算 y = alpha * op(A) * x + beta * y
	result := make([]float64, len(y))
	for i := 0; i < len(y); i++ {
		sum := 0.0
		for j := 0; j < len(x); j++ {
			if trans {
				sum += get(j, i) * x[j]
			} else {
				sum += get(i, j) * x[j]
			}
		}
		result[i] = alpha*sum + beta*y[i]
	}
	copy(y, result)
}

func TestDgemm(t *testing.T) {

	A := []float64{
		1, 2, 3,
		4, 5, 6,
	}
	B := []float64{
		7, 8,
		9, 10,
		11, 12,
	}

	C1 := make([]float64, 4)
	C2 := make([]float64, 9)

	dgemm(2, 2, 3, 1.0, A, false, B, false, 0.0, C1)
	if E1 := []float64{58, 64, 139, 154}; !reflect.DeepEqual(C1, E1) {
		t.Fatal("GEMM Test failed! Expected:", E1, "Got:", C1)
	}

	dgemm(3, 3, 2, 1.0, A, true, B, true, 0.0, C2)
	if E2 := []float64{39, 49, 59, 54, 68, 82, 69, 87, 105}; !reflect.DeepEqual(C2, E2) {
		t.Fatal("GEMM Test failed! Expected:", E2, "Got:", C2)
	}

	dgemm(3, 3, 2, 1.0, B, false, A, false, 0.0, C2)
	if E2 := []float64{39, 54, 69, 49, 68, 87, 59, 82, 105}; !reflect.DeepEqual(C2, E2) {
		t.Fatal("GEMM Test failed! Expected:", E2, "Got:", C2)
	}

	dgemm(2, 2, 3, 1.0, B, true, A, true, 0.0, C1)
	if E1 := []float64{58, 139, 64, 154}; !reflect.DeepEqual(C1, E1) {
		t.Fatal("GEMM Test failed! Expected:", E1, "Got:", C1)
	}

}

func dgemm(m, n, k int, alpha float64, A []float64, transA bool, B []float64, transB bool, beta float64, C []float64) {
	if len(A) != m*k || len(B) != k*n || len(C) != m*n {
		panic("bad shape")
	}

	get := func(data []float64, rows, cols, i, j int, transpose bool) float64 {
		if transpose {
			return data[j*rows+i]
		}
		return data[i*cols+j]
	}

	result := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0
			for p := 0; p < k; p++ {
				a := get(A, m, k, i, p, transA)
				b := get(B, k, n, p, j, transB)
				sum += a * b
			}
			result[i*n+j] = alpha*sum + beta*C[i*n+j]
		}
	}

	copy(C, result)
}

func almostEqual[T float64 | []float64](a, b T, tol float64) bool {
	equalWithinAbs := func(a, b float64) bool {
		return a == b || math.Abs(a-b) <= tol
	}
	switch reflect.TypeFor[T]().Kind() {
	case reflect.Float64:
		return equalWithinAbs(any(a).(float64), any(b).(float64))
	case reflect.Slice:
		a, b := any(a).([]float64), any(b).([]float64)
		if len(a) != len(b) {
			return false
		}
		for i, a := range a {
			if !equalWithinAbs(a, b[i]) {
				return false
			}
		}
		return true
	default:
		panic("unknown type")
	}
}
