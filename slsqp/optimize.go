// Copyright ©2025 curioloop. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slsqp

import (
	"errors"
	"fmt"
	"math"
	"slices"
)

// Bound represents the bounds for an optimization variable.
type Bound struct {
	Lower, Upper float64
}

// Evaluation evaluate the function and derivative for objective and constraints.
//   - 𝒇(𝐱) : ℝⁿ → ℝ
//   - 𝒄(𝐱) : ℝⁿ → ℝᵐ
//   - 𝒇′(𝐱) : ℝⁿ → ℝⁿ (partials of the objective function)
//   - 𝒄′(𝐱) : ℝⁿ → ℝᵐˣⁿ (constraint normals)
type Evaluation func(x []float64, g []float64) (f float64)

// Termination specifies the stopping criteria for the optimization algorithm.
type Termination struct {
	// The norm accuracy that determines the final solution.
	Accuracy float64
	// The iteration stop when the number of iteration exceeds limit.
	MaxIterations int
	// The maximum number of iterations in the NNLS problem.
	NNLSIterations int
	// The iteration will stop when |𝒇ₖ| < 𝚏𝚝𝚘𝚕
	FEvalTolerance float64
	// The iteration will stop when |𝒇ₖ₊₁ - 𝒇ₖ| < 𝚍𝚏𝚝𝚘𝚕
	FDiffTolerance float64
	// The iteration will stop when |𝐱ₖ₊₁ - 𝐱ₖ| < 𝚍𝚡𝚝𝚘𝚕
	XDiffTolerance float64
}

// LineSearch specifies the options for the line-search.
type LineSearch struct {
	// if Exact is true then an exact line-search is performed,
	// otherwise an armijo-type line-search is used
	Exact bool
	// The step range for line-search: 0 < Alpha[Lower] < Alpha[Upper] ≤ 1
	Alpha *Bound
}

// Problem specifies the problem for SLSQP optimizer.
type Problem struct {
	N       int          // The problem dimension
	Stop    Termination  // Stop condition
	Line    LineSearch   // LineSearch option
	Object  Evaluation   // Objective function 𝒇(𝐱) and gradients 𝒇′(𝐱)
	EqCons  []Evaluation // Equality constraints 𝒄(𝐱) = 0 and normals 𝒄′(𝐱)
	NeqCons []Evaluation // Inequality constraints 𝒄(𝐱) ≥ 0 and normals 𝒄′(𝐱)
	Bounds  []Bound      // Optional bounds
	// Infinity for bounds:
	//  - lower bounds are considered not exist when 𝒍ᵢ ≤ - BndInf
	//  - upper bounds are considered not exist when 𝒖ᵢ ≥ BndInf
	BndInf float64
}

// New creates a new SLSQP optimizer for given problem.
func (p *Problem) New() (optimizer *Optimizer, err error) {

	obj, eq, neq, stop, line := p.Object, p.EqCons, p.NeqCons, p.Stop, p.Line
	n, m, meq := p.N, len(eq)+len(neq), len(eq)

	inf := math.Abs(p.BndInf)
	bnd := p.Bounds

	if bnd == nil {
		bnd = make([]Bound, n)
		for i := range bnd {
			bnd[i].Upper = math.Inf(1)
			bnd[i].Lower = math.Inf(-1)
		}
	}

	if p.BndInf == zero {
		inf = math.MaxFloat64
	}

	const alfmin = 0.1
	if line.Alpha == nil {
		line.Alpha = &Bound{alfmin, one}
	} else {
		alpha := *line.Alpha
		if math.IsNaN(alpha.Lower) {
			alpha.Lower = alfmin
		}
		if math.IsNaN(alpha.Upper) {
			alpha.Upper = one
		}
		line.Alpha = &alpha
	}

	switch {
	case n <= 0:
		err = errors.New("problem dimension must greater than 0")
	case meq > n:
		err = errors.New("equality constrains number must not greater than n")
	case obj == nil:
		err = errors.New("objective function is required")
	case stop.MaxIterations <= 0:
		err = errors.New("max iteration must greater than 1")
	case stop.NNLSIterations < 0:
		err = errors.New("nnls iteration must not less than 0")
	case stop.Accuracy <= zero:
		err = errors.New("solution accuracy must not less than 0")
	case !math.IsNaN(stop.FEvalTolerance) && stop.FDiffTolerance < zero:
		err = errors.New("function eval tolerance must not less than 0")
	case !math.IsNaN(stop.FDiffTolerance) && stop.FDiffTolerance < zero:
		err = errors.New("function diff tolerance must not less than 0")
	case !math.IsNaN(stop.XDiffTolerance) && stop.XDiffTolerance < zero:
		err = errors.New("location diff tolerance must not less than 0")
	case line.Alpha.Lower < zero || line.Alpha.Upper > one || line.Alpha.Upper < line.Alpha.Lower:
		err = errors.New("line search alpha error")
	case len(bnd) != n:
		err = errors.New("bound size must equal to n")
	}

	for k, c := range eq {
		if c == nil {
			err = errors.New(fmt.Sprintf("equality constraint error at %d", k))
			break
		}
	}
	for k, c := range neq {
		if c == nil {
			err = errors.New(fmt.Sprintf("inequality constraint error at %d", k))
			break
		}
	}

	bnd = slices.Repeat(bnd, 1)
	for k, b := range bnd {
		if math.IsInf(b.Lower, 0) {
			b.Lower = math.NaN()
		}
		if math.IsInf(b.Upper, 0) {
			b.Upper = math.NaN()
		}
		l, u := !math.IsNaN(b.Lower), !math.IsNaN(b.Upper)
		if l && u && b.Lower > b.Upper {
			err = errors.New(fmt.Sprintf("bound error at %d", k))
			break
		}
	}

	if err != nil {
		return
	}

	optimizer = &Optimizer{
		sqpSpec{
			n: n, m: m, meq: meq,
			Problem: Problem{
				N:       n,
				Stop:    stop,
				Line:    line,
				Object:  obj,
				EqCons:  slices.Repeat(eq, 1),
				NeqCons: slices.Repeat(neq, 1),
				Bounds:  slices.Repeat(bnd, 1),
				BndInf:  inf,
			},
		},
	}

	return
}

// Optimizer implemented using the SLSQP algorithm.
type Optimizer struct {
	sqpSpec
}

// Workspace contains the state and context of the optimization process.
// Given problem dimension n and corrections number m,
// total work space is approximately float64[2×mn + 11×m² + 5×n + 8×m].
type Workspace struct {
	n, m, meq int
	sqpCtx
}

// Result contains the final result of the optimization process.
type Result struct {
	OK      bool      // Whether the optimization was converged.
	F       float64   // Final function value.
	X, G    []float64 // Final solution and gradient.
	Summary           // Optimization summary.
}

// Summary contains a summary of the optimization process.
type Summary struct {
	Status  sqpMode // Final task status after optimization.
	NumIter int     // Number of iterations performed.
}

// Init allocate the workspace for SLSQP optimizer.
// To avoid race conditions, separate workspaces need to be created for each goroutine.
// But multiple workspaces could share one optimizer.
func (o *Optimizer) Init() *Workspace {
	w := new(Workspace)
	w.n, w.m, w.meq = o.n, o.m, o.meq

	n, m, meq, n1 := w.n, w.m, w.meq, w.n+1
	mineq := (m - meq) + 2*n1
	totwk := /*LSQ*/ n1*(n1+1) + meq*(n1+1) + mineq*(n1+1) +
		/*LSI*/ (n1-meq+1)*(mineq+2) + 2*mineq +
		/*LSEI*/ (n1+mineq)*(n1-meq) + 2*meq + n1 +
		/*SLSQP*/ n1*n/2 + 2*m + 3*n + 3*n1 + 1
	wrk := make([]float64, totwk)

	la := max(1, m)
	ll := (n + 1) * (n + 2) / 2
	lr := n + n + m + 2

	im := 0
	il := im + la
	ix := il + n1*n/2 + 1
	ir := ix + n
	is := ir + n + n + la

	w.sqpCtx = sqpCtx{
		r:  wrk[ir : ir+lr], // r overlaps s  : (m + 2) - max(1, m)
		l:  wrk[il : il+ll], // l overlaps x0 : n
		x0: wrk[ix : ix+n],
		mu: wrk[im : im+la],
		s:  wrk[is : is+n1*1],
		u:  wrk[is+n1*1 : is+n1*2],
		v:  wrk[is+n1*2 : is+n1*3],
		w:  wrk[is+n1*3:],
		jw: make([]int, max(mineq, n1-mineq)),
	}

	return w
}

// Fit runs the optimization process using the initial guess x and workspace w.
func (o *Optimizer) Fit(x []float64, w *Workspace) *Result {

	if len(x) != o.n {
		panic("initial x dimension not match spec")
	}

	if w.n != o.n || w.m != o.m || w.meq != o.meq {
		panic("workspace dimension not match spec")
	}

	la := max(1, o.m)
	loc := sqpLoc{
		x: slices.Repeat(x, 1),
		g: make([]float64, o.n+1),
		c: make([]float64, la),
		a: make([]float64, la*(o.n+1)),
	}

	solver := sqpSolver{
		optimizer: o,
		workspace: w,
		location:  &loc,
	}

	res := solver.mainLoop()
	return &Result{
		OK: res == OK,
		X:  loc.x, F: loc.f, G: loc.g,
		Summary: Summary{
			Status:  res,
			NumIter: w.iter,
		},
	}
}
