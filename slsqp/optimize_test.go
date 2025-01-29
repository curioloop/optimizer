// Copyright ©2025 curioloop. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slsqp

import (
	"math"
	"testing"
)

// Case Sources : https://github.com/jacobwilliams/slsqp/blob/master/test/slsqp_test.f90
func TestRosenbrock(t *testing.T) {

	const n = 2

	objective := Evaluation{
		Function: func(x []float64) float64 {
			return 100.0*math.Pow(x[1]-math.Pow(x[0], 2), 2) + math.Pow(1.0-x[0], 2)
		},
		Derivative: func(x []float64, d []float64) {
			d[0] = -400.0*(x[1]-math.Pow(x[0], 2))*x[0] - 2.0*(1.0-x[0]) // ∂f/∂x1
			d[1] = 200.0 * (x[1] - math.Pow(x[0], 2))                    // ∂f/∂x2
		},
	}
	constraint := Evaluation{
		Function: func(x []float64) float64 {
			return 1.0 - math.Pow(x[0], 2) - math.Pow(x[1], 2)
		},
		Derivative: func(x []float64, d []float64) {
			d[0] = -2.0 * x[0] // ∂c/∂x1
			d[1] = -2.0 * x[1] // ∂c/∂x2
		},
	}

	x := []float64{0.1, 0.1}

	bounds := []Bound{
		{-1, 1},
		{-1, 1},
	}

	stop := Termination{
		Accuracy:      1e-8,
		MaxIterations: 50,
	}

	p := Problem{
		N:      n,
		Object: objective,
		NeqCons: []Evaluation{
			constraint,
		},
		Stop:   stop,
		Bounds: bounds,
	}

	s, e := p.New()
	if e != nil {
		panic(e)
	}
	w := s.Init()
	r := s.Fit(x, w)

	wantX := []float64{0.7864151509718389, 0.6176983165954114}
	wantF := 0.0456748087191604

	switch {
	case !r.OK:
		t.Fatal("TestRosenbrock: Not Converge")
	case r.F > wantF:
		t.Fatal("TestRosenbrock: Object Too Large")
	case !almostEqual(r.X, wantX, 1e-10):
		t.Fatal("TestRosenbrock: Bad Solution")
	case r.NumIter > 12:
		t.Fatal("TestRosenbrock: Too Many Iterations")
	}

}

// Case Sources : https://github.com/jacobwilliams/slsqp/blob/master/test/slsqp_test_2.f90
// (modfied with linesearch_mode = 2)
func TestBasic(t *testing.T) {

	const n = 3

	objective := Evaluation{
		Function: func(x []float64) float64 {
			return x[0]*x[0] + x[1]*x[1] + x[2]
		},
		Derivative: func(x []float64, d []float64) {
			d[0], d[1], d[2] = 2*x[0], 2*x[1], 1
		},
	}
	equality := Evaluation{
		Function: func(x []float64) float64 {
			return x[0]*x[1] - x[2]
		},
		Derivative: func(x []float64, d []float64) {
			d[0], d[1], d[2] = x[1], x[0], -1
		},
	}
	inequality := Evaluation{
		Function: func(x []float64) float64 {
			return x[2] - 1
		},
		Derivative: func(x []float64, d []float64) {
			d[0], d[1], d[2] = 0, 0, 1
		},
	}

	x := []float64{1, 2, 3}

	bounds := []Bound{
		{-10, 10},
		{-10, 10},
		{-10, 10},
	}

	stop := Termination{
		Accuracy:      1e-7,
		MaxIterations: 50,
	}

	line := LineSearch{
		Alpha: &Bound{Lower: 0.1, Upper: 0.5},
	}

	{

		p := Problem{
			N:       n,
			Object:  objective,
			EqCons:  []Evaluation{equality},
			NeqCons: []Evaluation{inequality},
			Line:    line,
			Stop:    stop,
			Bounds:  bounds,
		}

		s, e := p.New()
		if e != nil {
			panic(e)
		}
		w := s.Init()
		r := s.Fit(x, w)

		wantX := []float64{1.0000000213633444, 0.99999997029033982, 1.0000000596046676}
		wantF := 3.0000000429120375

		switch {
		case !r.OK:
			t.Fatal("TestBasic: Not Converge")
		case r.F > wantF:
			t.Fatal("TestBasic: Object Too Large")
		case !almostEqual(r.X, wantX, 1e-10):
			t.Fatal("TestBasic: Bad Solution")
		case r.NumIter > 25:
			t.Fatal("TestBasic: Too Many Iterations")
		}

	}

	line.Exact = true

	{
		p := Problem{
			N:       n,
			Object:  objective,
			EqCons:  []Evaluation{equality},
			NeqCons: []Evaluation{inequality},
			Line:    line,
			Stop:    stop,
			Bounds:  bounds,
		}

		s, e := p.New()
		if e != nil {
			panic(e)
		}
		w := s.Init()
		r := s.Fit(x, w)

		wantX := []float64{0.9999999945350062, 0.9999999945376942, 1.0000000596086882}
		wantF := 3.0000000377540887

		switch {
		case !r.OK:
			t.Fatal("TestBasic: Not Converge")
		case r.F > wantF:
			t.Fatal("TestBasic: Object Too Large")
		case !almostEqual(r.X, wantX, 1e-10):
			t.Fatal("TestBasic: Bad Solution")
		case r.NumIter > 25:
			t.Fatal("TestBasic: Too Many Iterations")
		}
	}

}

// Case Sources : https://github.com/jacobwilliams/slsqp/blob/master/test/slsqp_test_71.f90
func TestProb71(t *testing.T) {

	const n = 5

	obj := Evaluation{
		Function: func(x []float64) float64 {
			return x[0]*x[3]*(x[0]+x[1]+x[2]) + x[2]
		},
		Derivative: func(x []float64, d []float64) {
			d[0] = x[3] * (2.0*x[0] + x[1] + x[2])
			d[1] = x[0] * x[3]
			d[2] = x[0]*x[3] + 1.0
			d[3] = x[0] * (x[0] + x[1] + x[2])
			d[4] = 0.0
		},
	}
	cons1 := Evaluation{
		Function: func(x []float64) float64 {
			return x[0]*x[1]*x[2]*x[3] - x[4] - 25
		},
		Derivative: func(x []float64, d []float64) {
			d[0] = x[1] * x[2] * x[3]
			d[1] = x[0] * x[2] * x[3]
			d[2] = x[0] * x[1] * x[3]
			d[3] = x[0] * x[1] * x[2]
			d[4] = -1
		},
	}
	cons2 := Evaluation{
		Function: func(x []float64) float64 {
			return x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3] - 40
		},
		Derivative: func(x []float64, d []float64) {
			d[0] = 2 * x[0]
			d[1] = 2 * x[1]
			d[2] = 2 * x[2]
			d[3] = 2 * x[3]
			d[4] = 0
		},
	}

	x := []float64{1, 5, 5, 1, -24}

	bounds := []Bound{
		{1, 5},
		{1, 5},
		{1, 5},
		{1, 5},
		{0, 1e10},
	}

	stop := Termination{
		Accuracy:      1e-8,
		MaxIterations: 50,
	}

	p := Problem{
		N:      n,
		Object: obj,
		EqCons: []Evaluation{cons1, cons2},
		Stop:   stop,
		Bounds: bounds,
	}

	s, e := p.New()
	if e != nil {
		panic(e)
	}
	w := s.Init()
	r := s.Fit(x, w)

	wantX := []float64{1, 4.7429996586260321, 3.8211499562762130, 1.3794082970345380, 0}
	wantF := 17.0140172891520542

	switch {
	case !r.OK:
		t.Fatal("TestProb71: Not Converge")
	case r.F > wantF:
		t.Fatal("TestProb71: Object Too Large")
	case !almostEqual(r.X, wantX, 1e-10):
		t.Fatal("TestProb71: Bad Solution")
	case r.NumIter > 12:
		t.Fatal("TestProb71: Too Many Iterations")
	}

}

// Case Sources : https://github.com/scipy/scipy/blob/main/scipy/optimize/tests/test_slsqp.py (test_inconsistent_linearization)
func TestBadCase(t *testing.T) {

	const n = 2

	objective := Evaluation{
		Function: func(x []float64) float64 {
			return x[0]*x[0] + x[1]*x[1]
		},
		Derivative: func(x []float64, d []float64) {
			d[0], d[1] = 2*x[0], 2*x[1]
		},
	}
	equality := Evaluation{
		Function: func(x []float64) float64 {
			return x[0] + x[1] - 2
		},
		Derivative: func(x []float64, d []float64) {
			d[0], d[1] = 1, 1
		},
	}
	inequality := Evaluation{
		Function: func(x []float64) float64 {
			return x[0]*x[0] - 1
		},
		Derivative: func(x []float64, d []float64) {
			d[0], d[1] = 2*x[0], 0
		},
	}

	x := []float64{0, 1}

	bounds := []Bound{
		{0, math.NaN()},
		{0, math.NaN()},
	}

	stop := Termination{
		Accuracy:      1e-6,
		MaxIterations: 50,
	}

	wantX := []float64{1, 1}

	{
		p := Problem{
			N:       n,
			Object:  objective,
			EqCons:  []Evaluation{equality},
			NeqCons: []Evaluation{inequality},
			Stop:    stop,
			Bounds:  bounds,
		}

		s, e := p.New()
		if e != nil {
			panic(e)
		}
		w := s.Init()
		r := s.Fit(x, w)

		switch {
		case !r.OK:
			t.Fatal("TestBad: Not Converge")
		case math.Abs(equality.Function(r.X)) > 1e15:
			t.Fatal("TestBad: EqCons Violation")
		case inequality.Function(r.X) < zero:
			t.Fatal("TestBad: NeqCons Violation")
		case !almostEqual(r.X, wantX, 1e-10):
			t.Fatal("TestBad: Bad Solution")
		case r.NumIter > 3:
			t.Fatal("TestBad: Too Many Iterations")
		}

	}

	bounds = []Bound{
		{0, 0},
		{math.NaN(), math.NaN()},
	}

	{
		p := Problem{
			N:       n,
			Object:  objective,
			EqCons:  []Evaluation{equality},
			NeqCons: []Evaluation{inequality},
			Stop:    stop,
			Bounds:  bounds,
		}

		s, e := p.New()
		if e != nil {
			panic(e)
		}
		w := s.Init()
		r := s.Fit(x, w)

		switch {
		case r.OK || r.Status != SearchNotDescent:
			t.Fatal("TestBad: Unexpected Status")
		case r.NumIter > 5:
			t.Fatal("TestBad: Too Many Iterations")
		}

	}

}

// Case Sources : https://github.com/scipy/scipy/blob/main/scipy/optimize/tests/test_slsqp.py (test_bounds_clipping)
func TestBoundClip(t *testing.T) {

	const n = 1

	obj := Evaluation{
		Function: func(x []float64) float64 {
			return (x[0] - 1) * (x[0] - 1)
		},
		Derivative: func(x []float64, d []float64) {
			d[0] = 2*x[0] - 2
		},
	}

	stop := Termination{
		Accuracy:      1e-6,
		MaxIterations: 50,
	}

	tests := []struct {
		init    float64
		bnd     []Bound
		desired float64
	}{
		{10, []Bound{{math.NaN(), 0}}, 0},
		{-10, []Bound{{2, math.NaN()}}, 2},
		{-10, []Bound{{math.NaN(), 0}}, 0},
		{10, []Bound{{2, math.NaN()}}, 2},
		{-0.5, []Bound{{-1, 0}}, 0},
		{10, []Bound{{-1, 0}}, 0},
	}

	for _, tt := range tests {

		p := Problem{
			N:      n,
			Object: obj,
			Bounds: tt.bnd,
			Stop:   stop,
		}

		s, e := p.New()
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

// Case Sources : https://github.com/scipy/scipy/blob/main/scipy/optimize/tests/test_slsqp.py (test_infeasible_initial)
func TestInfeasibleInit(t *testing.T) {

	const n = 1

	obj := Evaluation{
		Function: func(x []float64) float64 {
			return x[0]*x[0] - 2*x[0] + 1
		},
		Derivative: func(x []float64, d []float64) {
			d[0] = 2*x[0] - 2
		},
	}

	stop := Termination{
		Accuracy:      1e-6,
		MaxIterations: 50,
	}

	consU := []Evaluation{{
		Function:   func(x []float64) float64 { return 0 - x[0] },
		Derivative: func(x []float64, d []float64) { d[0] = -1 },
	}}
	consL := []Evaluation{{
		Function:   func(x []float64) float64 { return x[0] - 2 },
		Derivative: func(x []float64, d []float64) { d[0] = 1 },
	}}
	consUL := []Evaluation{{
		Function:   func(x []float64) float64 { return 0 - x[0] },
		Derivative: func(x []float64, d []float64) { d[0] = -1 },
	}, {
		Function:   func(x []float64) float64 { return x[0] + 1 },
		Derivative: func(x []float64, d []float64) { d[0] = 1 },
	}}

	tests := []struct {
		init []float64
		cons []Evaluation
	}{
		{[]float64{10}, consU},
		{[]float64{-10}, consL},
		{[]float64{-10}, consU},
		{[]float64{10}, consL},
		{[]float64{-0.5}, consUL},
		{[]float64{10}, consUL},
	}

	for _, tt := range tests {

		p := Problem{
			N:       n,
			Object:  obj,
			NeqCons: tt.cons,
			Stop:    stop,
		}

		s, e := p.New()
		if e != nil {
			panic(e)
		}
		w := s.Init()
		r := s.Fit(tt.init, w)

		switch {
		case !r.OK:
			t.Fatal("TestInfeasibleInit: Not Converge")
		case math.Abs(r.X[0]) > 1e15:
			t.Fatal("TestInfeasibleInit: EqCons Violation")
		}

	}

}

// Case Sources : https://github.com/scipy/scipy/blob/main/scipy/optimize/tests/test_slsqp.py (test_inconsistent_inequalities)
func TestInconsistentCons(t *testing.T) {

	const n = 2

	obj := Evaluation{
		Function: func(x []float64) float64 {
			return -1*x[0] + 4*x[1]
		},
		Derivative: func(x []float64, d []float64) {
			d[0], d[1] = -1, 4
		},
	}
	cons1 := Evaluation{
		Function: func(x []float64) float64 {
			return x[1] - x[0] - 1
		},
		Derivative: func(x []float64, d []float64) {
			d[0], d[1] = -1, 1
		},
	}
	cons2 := Evaluation{
		Function: func(x []float64) float64 {
			return x[0] - x[1]
		},
		Derivative: func(x []float64, d []float64) {
			d[0], d[1] = 1, -1
		},
	}

	x := []float64{1, 5}

	bounds := []Bound{
		{-5, 5},
		{-5, 5},
	}

	stop := Termination{
		Accuracy:      1e-6,
		MaxIterations: 50,
	}

	p := Problem{
		N:       n,
		Object:  obj,
		NeqCons: []Evaluation{cons1, cons2},
		Stop:    stop,
		Bounds:  bounds,
	}

	s, e := p.New()
	if e != nil {
		panic(e)
	}
	w := s.Init()
	r := s.Fit(x, w)

	switch {
	case r.OK || r.Status != SearchNotDescent:
		t.Fatal("TestInconsistentCons: Unexpected Status")
	case r.NumIter > 11:
		t.Fatal("TestInconsistentCons: Too Many Iterations")
	}
}

// Case Sources : https://github.com/scipy/scipy/blob/main/scipy/optimize/tests/test_slsqp.py (test_gh1758)
func TestSqrtR2(t *testing.T) {

	// TODO: this case not pass yet, but its behavior is consistent with the original version.

	const n = 2

	obj := Evaluation{
		Function: func(x []float64) float64 {
			return math.Sqrt(x[1])
		},
		Derivative: func(x []float64, d []float64) {
			d[0], d[1] = 0, 0.5/math.Sqrt(x[1])
		},
	}
	cons1 := Evaluation{
		Function: func(x []float64) float64 {
			xx := 2 * x[0]
			return x[1] - xx*xx*xx
		},
		Derivative: func(x []float64, d []float64) {
			xx := 2 * x[0]
			d[0], d[1] = -6*xx*xx, 1
		},
	}
	cons2 := Evaluation{
		Function: func(x []float64) float64 {
			xx := -x[0] + 1
			return x[1] - xx*xx*xx
		},
		Derivative: func(x []float64, d []float64) {
			xx := -x[0] + 1
			d[0], d[1] = -3*xx*xx, 1
		},
	}

	x := []float64{8, 0.25}

	bounds := []Bound{
		{-0.5, 1},
		{0, 8},
	}

	stop := Termination{
		Accuracy:      1e-6,
		MaxIterations: 50,
	}

	p := Problem{
		N:      n,
		Object: obj,
		EqCons: []Evaluation{cons1, cons2},
		Stop:   stop,
		Bounds: bounds,
	}

	s, e := p.New()
	if e != nil {
		panic(e)
	}
	w := s.Init()
	r := s.Fit(x, w)

	switch {
	case r.OK || r.Status != SQPExceedMaxIter:
		t.Fatal("TestSqrtR2: Unexpected Status")
	case r.NumIter != 50:
		t.Fatal("TestSqrtR2: Too Many Iterations")
	}
}
