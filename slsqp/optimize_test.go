// Copyright ©2025 curioloop. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slsqp

import (
	"github.com/curioloop/optimizer/numdiff"
	"math"
	"testing"
)

// Case Sources : https://github.com/jacobwilliams/slsqp/blob/master/test/slsqp_test.f90
func TestRosenbrock(t *testing.T) {

	const n = 2

	objective := func(x []float64, g []float64) (f float64) {
		if g == nil {
			f = 100.0*math.Pow(x[1]-math.Pow(x[0], 2), 2) + math.Pow(1.0-x[0], 2)
		} else {
			g[0] = -400.0*(x[1]-math.Pow(x[0], 2))*x[0] - 2.0*(1.0-x[0]) // ∂f/∂x1
			g[1] = 200.0 * (x[1] - math.Pow(x[0], 2))                    // ∂f/∂x2
		}
		return
	}

	constraint := func(x []float64, g []float64) (f float64) {
		if g == nil {
			f = 1.0 - math.Pow(x[0], 2) - math.Pow(x[1], 2)
		} else {
			g[0] = -2.0 * x[0] // ∂c/∂x1
			g[1] = -2.0 * x[1] // ∂c/∂x2
		}
		return
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

	objective := func(x []float64, g []float64) (f float64) {
		if g == nil {
			f = x[0]*x[0] + x[1]*x[1] + x[2]
		} else {
			g[0], g[1], g[2] = 2*x[0], 2*x[1], 1
		}
		return
	}

	equality := func(x []float64, g []float64) (f float64) {
		if g == nil {
			f = x[0]*x[1] - x[2]
		} else {
			g[0], g[1], g[2] = x[1], x[0], -1
		}
		return
	}

	inequality := func(x []float64, g []float64) (f float64) {
		if g == nil {
			f = x[2] - 1
		} else {
			g[0], g[1], g[2] = 0, 0, 1
		}
		return
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

	obj := func(x []float64, g []float64) (f float64) {
		if g == nil {
			f = x[0]*x[3]*(x[0]+x[1]+x[2]) + x[2]
		} else {
			g[0] = x[3] * (2.0*x[0] + x[1] + x[2])
			g[1] = x[0] * x[3]
			g[2] = x[0]*x[3] + 1.0
			g[3] = x[0] * (x[0] + x[1] + x[2])
			g[4] = 0.0
		}
		return
	}
	cons1 := func(x []float64, g []float64) (f float64) {
		if g == nil {
			f = x[0]*x[1]*x[2]*x[3] - x[4] - 25
		} else {
			g[0] = x[1] * x[2] * x[3]
			g[1] = x[0] * x[2] * x[3]
			g[2] = x[0] * x[1] * x[3]
			g[3] = x[0] * x[1] * x[2]
			g[4] = -1
		}
		return
	}
	cons2 := func(x []float64, g []float64) (f float64) {
		if g == nil {
			f = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3] - 40
		} else {
			g[0] = 2 * x[0]
			g[1] = 2 * x[1]
			g[2] = 2 * x[2]
			g[3] = 2 * x[3]
			g[4] = 0
		}
		return
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

	objective := func(x []float64, g []float64) (f float64) {
		if g == nil {
			f = x[0]*x[0] + x[1]*x[1]
		} else {
			g[0], g[1] = 2*x[0], 2*x[1]
		}
		return
	}

	equality := func(x []float64, g []float64) (f float64) {
		if g == nil {
			f = x[0] + x[1] - 2
		} else {
			g[0], g[1] = 1, 1
		}
		return
	}

	inequality := func(x []float64, g []float64) (f float64) {
		if g == nil {
			f = x[0]*x[0] - 1
		} else {
			g[0], g[1] = 2*x[0], 0
		}
		return
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
		case math.Abs(equality(r.X, nil)) > 1e15:
			t.Fatal("TestBad: EqCons Violation")
		case inequality(r.X, nil) < zero:
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

	obj := func(x []float64, g []float64) (f float64) {
		if g == nil {
			f = (x[0] - 1) * (x[0] - 1)
		} else {
			g[0] = 2*x[0] - 2
		}
		return
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

	obj := func(x []float64, g []float64) (f float64) {
		if g != nil {
			g[0] = 2*x[0] - 2
		}
		return x[0]*x[0] - 2*x[0] + 1
	}
	consU := []Evaluation{func(x []float64, g []float64) (f float64) {
		if g != nil {
			f = 0 - x[0]
		}
		return 0 - x[0]
	}}
	consL := []Evaluation{func(x []float64, g []float64) (f float64) {
		if g != nil {
			g[0] = 1
		}
		return x[0] - 2
	}}
	consUL := []Evaluation{func(x []float64, g []float64) (f float64) {
		if g != nil {
			g[0] = -1
		}
		return 0 - x[0]
	}, func(x []float64, g []float64) (f float64) {
		if g != nil {
			g[0] = 1
		}
		return x[0] + 1
	}}

	stop := Termination{
		Accuracy:      1e-6,
		MaxIterations: 50,
	}

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

	obj := func(x []float64, g []float64) (f float64) {
		if g == nil {
			f = -1*x[0] + 4*x[1]
		} else {
			g[0], g[1] = -1, 4
		}
		return
	}

	cons1 := func(x []float64, g []float64) (f float64) {
		if g == nil {
			f = x[1] - x[0] - 1
		} else {
			g[0], g[1] = -1, 1
		}
		return
	}

	cons2 := func(x []float64, g []float64) (f float64) {
		if g == nil {
			f = x[0] - x[1]
		} else {
			g[0], g[1] = 1, -1
		}
		return
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

	eps := math.Sqrt(eps)
	bnd := []numdiff.Bound{{-0.5, 1}, {0, 8}}
	wrap := func(fun func([]float64) float64) func(x []float64, g []float64) (f float64) {
		app := numdiff.ApproxSpec{
			N: 2, M: 1, Bounds: bnd, AbsStep: eps, NotChkBnd: true,
			Object: func(x, y []float64) { y[0] = fun(x) }}
		return func(x []float64, g []float64) (f float64) {
			if g == nil {
				f = fun(x)
			} else {
				if err := app.Diff(x, g); err != nil {
					panic(err)
				}
			}
			return
		}
	}

	obj := wrap(func(x []float64) float64 {
		return math.Sqrt(x[1])
	})
	cons1 := wrap(func(x []float64) float64 {
		return x[1] - math.Pow(2*x[0], 3)
	})
	cons2 := wrap(func(x []float64) float64 {
		return x[1] - math.Pow(-x[0]+1, 3)
	})

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
		N:      2,
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

	wantX := []float64{0.33333333333388, 0.29629629629483}
	wantF := 0.5443310539518

	switch {
	case !r.OK:
		t.Fatal("TestSqrtR2: Not Converge")
	case r.F > wantF:
		t.Fatal("TestSqrtR2: Object Too Large")
	case !almostEqual(r.X, wantX, 1e-10):
		t.Fatal("TestSqrtR2: Bad Solution")
	case r.NumIter > 8:
		t.Fatal("TestSqrtR2: Too Many Iterations")
	}
}
