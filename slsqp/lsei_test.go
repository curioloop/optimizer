// Copyright ©2025 curioloop. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slsqp

import (
	"testing"
)

// C.L. Lawson, R.J. Hanson, 'Solving least squares problems' Prentice Hall, 1974. (revised 1995 edition)
// Chapters 23, Section 7.
func TestLSI(t *testing.T) {

	const (
		n  = 2
		me = 4
		mg = 3
		mc = 0
	)

	wantX := []float64{0.62131519274376423, 0.37868480725623571}
	wantW := []float64{0.0000000000000000, 0.0000000000000000, 0.21156462585034014}
	wantNorm := 0.33822934965866208

	{
		e := []float64{
			0.25, 0.5, 0.5, 0.8,
			1, 1, 1, 1,
		}
		f := []float64{0.5, 0.6, 0.7, 1.2}
		g := []float64{
			1, 0, -1,
			0, 1, -1,
		}
		h := []float64{0, 0, -1}

		x := make([]float64, n)
		w := make([]float64, (n+1)*(mg+2)+2*mg)
		jw := make([]int, mg)

		norm, mode := LSI(e, f, g, h, me, me, mg, mg, n, x, w, jw, 0)
		if mode != HasSolution {
			t.Fatal("LSI no solution")
		}
		if !almostEqual(wantNorm, norm, 1e-15) {
			t.Fatal("LSI residual norm error")
		}
		if !almostEqual(wantX, x, 1e-15) {
			t.Fatal("LSI solution unexpected")
		}
		if !almostEqual(wantW, w[:mg], 1e-15) {
			t.Fatal("LSI solution unexpected")
		}
	}

	{
		e := []float64{
			0.25, 0.5, 0.5, 0.8,
			1, 1, 1, 1,
		}
		f := []float64{0.5, 0.6, 0.7, 1.2}
		g := []float64{
			1, 0, -1,
			0, 1, -1,
		}
		h := []float64{0, 0, -1}

		x := make([]float64, n)
		w := make([]float64, 2*mc+me+(me+mg)*(n-mc)+(n-mc+1)*(mg+2)+2*mg)
		jw := make([]int, max(mg, min(me, n-mc)))

		norm, mode := LSEI(nil, nil, e, f, g, h, mc, mc, me, me, mg, mg, n, x, w, jw, 0)
		if mode != HasSolution {
			t.Fatal("LSI no solution")
		}
		if !almostEqual(wantNorm, norm, 1e-15) {
			t.Fatal("LSI residual norm error")
		}
		if !almostEqual(wantX, x, 1e-15) {
			t.Fatal("LSI solution unexpected")
		}
		if !almostEqual(wantW, w[:mc+mg], 1e-15) {
			t.Fatal("LSI solution unexpected")
		}

	}

}

// C.L. Lawson, R.J. Hanson, 'Solving least squares problems' Prentice Hall, 1974. (revised 1995 edition)
// Chapters 20.
func TestLSE(t *testing.T) {

	const (
		n  = 2
		me = 2
		mg = 0
		mc = 1
	)

	e := []float64{
		0.4302, 0.6246,
		0.3516, 0.3384,
	}
	f := []float64{
		0.6593, 0.9666,
	}
	c := []float64{
		0.4087,
		0.1593,
	}
	d := []float64{
		0.1376,
	}

	wantX := []float64{-1.1774989821678763, 3.8847698305838736}
	wantW := []float64{-0.38159188319253667}
	wantNorm := 0.43604479747076780

	x := make([]float64, n)
	w := make([]float64, 2*mc+me+(me+mg)*(n-mc)+(n-mc+1)*(mg+2)+2*mg)
	jw := make([]int, max(mg, min(me, n-mc)))

	norm, mode := LSEI(c, d, e, f, nil, nil, mc, mc, me, me, mg, mg, n, x, w, jw, 0)
	if mode != HasSolution {
		t.Fatal("LSE no solution")
	}
	if !almostEqual(wantNorm, norm, 1e-15) {
		t.Fatal("LSE residual norm error")
	}
	if !almostEqual(wantX, x, 1e-15) {
		t.Fatal("LSE solution unexpected")
	}
	if !almostEqual(wantW, w[:mc+mg], 1e-15) {
		t.Fatal("LSE solution unexpected")
	}

}

func TestLSEI(t *testing.T) {

	const (
		n  = 3
		me = 4
		mc = 2
		mg = 1
	)

	e := []float64{
		3, 1, 2, 0,
		2, 0, 0, 1,
		1, 0, 2, 0,
	}
	f := []float64{2, 1, 8, 3}
	g := []float64{
		0,
		1,
		0,
	}
	h := []float64{3}
	c := []float64{
		-1, 2,
		0, 1,
		0, -1,
	}
	d := []float64{-3, 2}

	wantX := []float64{3, 3, 7}
	wantW := []float64{-174, -44, 84}
	wantNorm := 23.769728648

	x := make([]float64, n)
	w := make([]float64, 2*mc+me+(me+mg)*(n-mc)+(n-mc+1)*(mg+2)+2*mg)
	jw := make([]int, max(mg, min(me, n-mc)))

	norm, mode := LSEI(c, d, e, f, g, h, mc, mc, me, me, mg, mg, n, x, w, jw, 0)
	if mode != HasSolution {
		t.Fatal("LSE no solution")
	}
	if !almostEqual(wantNorm, norm, 1e-10) {
		t.Fatal("LSE residual norm error")
	}
	if !almostEqual(wantX, x, 1e-10) {
		t.Fatal("LSE solution unexpected")
	}
	if !almostEqual(wantW, w[:mc+mg], 1e-10) {
		t.Fatal("LSE solution unexpected")
	}
}
