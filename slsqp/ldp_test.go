// Copyright ©2025 curioloop. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slsqp

import (
	"testing"
)

// Origin: https://www.netlib.org/lawson-hanson/all (PROG6)
// Reference: https://people.math.sc.edu/Burkardt/f_src/lawson/lawson.html
func TestLDP(t *testing.T) {

	const m = 3
	const n = 2

	g2 := []float64{
		0.20718533228468983, 0.39218501461672955, -0.59937034690141933,
		-2.5576231892137238, 1.3511531307082973, 1.2064700585054264,
	}

	h2 := []float64{
		-1.3004115226337452, -0.083539094650205481, 0.38395061728395063,
	}

	wantX := []float64{-0.12680556318798736, 0.25524638652733850}
	wantW := []float64{0.0000000000000000, 0.0000000000000000, 0.21156462585034014}
	wantNorm := 0.2850094185999581

	x := make([]float64, n)
	w := make([]float64, (n+1)*(m+2)+2*m)
	jw := make([]int, m)

	norm, mode := LDP(m, n, g2, m, h2, x, w, jw, 30)
	if mode != HasSolution {
		t.Fatal("LDP no solution")
	}
	if !almostEqual(wantNorm, norm, 1e-15) {
		t.Fatal("LDP residual norm error")
	}
	if !almostEqual(wantX, x, 1e-15) {
		t.Fatal("LDP solution unexpected")
	}
	if !almostEqual(wantW, w[:m], 1e-15) {
		t.Fatal("LDP solution unexpected")
	}
}
