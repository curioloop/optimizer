// Copyright ©2025 curioloop. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lbfgsb

import (
	"math"
	"os"
	"testing"
)

func TestBasic(t *testing.T) {

	K := []float64{1., 0.3, 0.5}
	F := []float64{
		1, 1, 1,
		1, 1, 0,
		1, 0, 1,
		1, 0, 0,
		1, 0, 0,
	}

	x0 := []float64{0, 0, 0}

	eval := func(x []float64, g []float64) (f float64) {
		const m = 5
		const n = 3
		Fx := make([]float64, 5)
		dgemv(m, n, 1, F, false, x, 0, Fx)
		sum := zero
		for _, v := range Fx {
			sum += math.Exp(v)
		}
		logZ := math.Log(sum)
		f = logZ - ddot(3, K, 1, x, 1)
		for i, v := range Fx {
			Fx[i] = math.Exp(v - logZ)
		}
		dgemm(n, 1, m, 1, F, true, Fx, false, 0, g)
		for i, k := range K {
			g[i] -= k
		}
		return
	}

	stop := Termination{
		MaxIterations:     10,
		MaxComputations:   10,
		MaxEvaluations:    10,
		EpsAccuracyFactor: 1e7,
		ProjGradTolerance: 1e-5,
	}

	f, _ := os.Open(os.DevNull)
	log := &Logger{
		Level: LogVerbose,
		Out:   f,
	}

	p := Problem{
		N: 3, M: 5,
		Eval: eval,
		Stop: stop,
	}
	s, e := p.New(log)
	if e != nil {
		panic(e)
	}

	w := s.Init()
	r := s.Fit(x0, w)

	switch {
	case !r.OK:
		t.Fatal("TestBasic: Not Converge")
	case r.F > 1.559132167348348:
		t.Fatal("TestBasic: Object Too Large")
	case r.NumIter > 4:
		t.Fatal("TestBasic: Too Many Iterations")
	case r.NumEval > 5:
		t.Fatal("TestBasic: Too Many Evaluations")
	}
}

func TestRosenbrock(t *testing.T) {

	const n = 25
	const m = 5

	// Initialize bounds and starting point
	x := make([]float64, n)
	bounds := make([]Bound, n)
	for i := 0; i < n; i++ {
		if (i+1)%2 == 1 { // Odd variables
			bounds[i].Lower = 1.0
			bounds[i].Upper = 100.0
		} else { // Even variables
			bounds[i].Lower = -100.0
			bounds[i].Upper = 100.0
		}
		x[i] = 3.0 // Starting point
	}

	stop := Termination{
		MaxIterations:     50,
		MaxComputations:   100,
		MaxEvaluations:    100,
		EpsAccuracyFactor: 1e7,
		ProjGradTolerance: 1e-5,
	}

	eval := func(x []float64, g []float64) (f float64) {
		f = 0.25 * math.Pow(x[0]-1.0, 2)
		for i := 1; i < n; i++ {
			f += math.Pow(x[i]-math.Pow(x[i-1], 2), 2)
		}
		f *= 4.0

		t1 := x[1] - math.Pow(x[0], 2)
		g[0] = 2.0*(x[0]-1.0) - 16.0*x[0]*t1
		for i := 1; i < n-1; i++ {
			t2 := t1
			t1 = x[i+1] - math.Pow(x[i], 2)
			g[i] = 8.0*t2 - 16.0*x[i]*t1
		}
		g[n-1] = 8.0 * t1
		return f
	}

	f, _ := os.Open(os.DevNull)
	logger := &Logger{
		Level: LogVerbose,
		Out:   f,
	}

	p := Problem{
		N: n, M: m,
		Eval:   eval,
		Stop:   stop,
		Bounds: bounds,
	}

	s, e := p.New(logger)
	if e != nil {
		panic(e)
	}
	w := s.Init()
	r := s.Fit(x, w)

	switch {
	case !r.OK:
		t.Fatal("TestRosenbrock: Not Converge")
	case r.F > 1.0834901e-9:
		t.Fatal("TestRosenbrock: Object Too Large")
	case r.NumIter > 23:
		t.Fatal("TestRosenbrock: Too Many Iterations")
	case r.NumEval > 28:
		t.Fatal("TestRosenbrock: Too Many Evaluations")
	}
}

// Case Sources : https://github.com/scipy/scipy/blob/main/scipy/optimize/tests/test_slsqp.py (test_bounds_clipping)
func TestBoundClip(t *testing.T) {

	const n = 1
	const m = 5

	eval := func(x []float64, g []float64) (f float64) {
		g[0] = 2*x[0] - 2
		return (x[0] - 1) * (x[0] - 1)
	}

	stop := Termination{
		MaxIterations:     50,
		MaxComputations:   100,
		MaxEvaluations:    100,
		EpsAccuracyFactor: 1e7,
		ProjGradTolerance: 1e-5,
	}

	f, _ := os.Open(os.DevNull)
	logger := &Logger{
		Level: LogVerbose,
		Out:   f,
	}

	tests := []struct {
		init    float64
		bnd     []Bound
		desired float64
	}{
		{10, []Bound{{Lower: math.NaN(), Upper: 0}}, 0},
		{-10, []Bound{{Lower: 2, Upper: math.NaN()}}, 2},
		{-10, []Bound{{Lower: math.NaN(), Upper: 0}}, 0},
		{10, []Bound{{Lower: 2, Upper: math.NaN()}}, 2},
		{-0.5, []Bound{{Lower: -1, Upper: 0}}, 0},
		{10, []Bound{{Lower: -1, Upper: 0}}, 0},
	}

	for _, tt := range tests {
		p := Problem{
			N: n, M: m,
			Eval:   eval,
			Stop:   stop,
			Bounds: tt.bnd,
		}

		s, e := p.New(logger)
		if e != nil {
			panic(e)
		}

		w := s.Init()
		r := s.Fit([]float64{tt.init}, w)

		switch {
		case !r.OK:
			t.Fatal("TestBoundClip: Not Converge")
		case !almostEqual(r.X[0], tt.desired, 1e15):
			t.Fatal("TestBoundClip: EqCons Violation")
		}
	}

}
