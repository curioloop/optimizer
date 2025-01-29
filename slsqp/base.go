// Copyright ©2025 curioloop. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slsqp

const (
	zero = 0.0
	one  = 1.0
	two  = 2.0
	four = 4.0
	ten  = 10.0
	hun  = 100.0
	eps  = float64(7)/3 - float64(4)/3 - 1.
)

type sqpMode int

const (
	OK sqpMode = iota
	// HasSolution problem solved successfully.
	HasSolution
	// BadArgument evaluation panic or input dimension unacceptable.
	BadArgument
	// NNLSExceedMaxIter more than max iterations for solving NNLS
	NNLSExceedMaxIter
	// ConsIncompatible inequality constraints incompatible
	ConsIncompatible
	// LSISingularE matrix E is not of full rank in LSI
	LSISingularE
	// LSEISingularC matrix C is not of full rank in LSEI
	LSEISingularC
	// HFTIRankDefect rank-deficient equality constraint in HFTI
	HFTIRankDefect
	// SearchNotDescent positive directional derivative for line-search
	SearchNotDescent
	// SQPExceedMaxIter more than max iterations in SQP
	SQPExceedMaxIter
)

const (
	// evalGrad evaluate derivatives for loc.g and loc.a
	evalGrad sqpMode = -1
	// evalFunc evaluate functions for loc.f and loc.c
	evalFunc sqpMode = -2
)

type sqpSpec struct {
	// the number of variables
	n int
	// the total number of constraints
	m int
	// the number of equality constraints
	meq int
	Problem
}

type sqpLoc struct {
	f float64
	x []float64 // n
	c []float64 // 𝚖𝚊𝚡(1,m)
	g []float64 // n+1
	a []float64 // 𝚖𝚊𝚡(1,m) × (n+1)
}

type sqpCtx struct {
	// solution accuracy for convergence.
	acc float64
	// relaxed tolerance for convergence.
	tol float64
	// line-search initial value of objective function.
	f0 float64
	// line-search initial value of merit function.
	t0 float64
	// line-search step length.
	alpha float64
	// line-search counter.
	line int
	// iteration counter.
	iter int
	// BFGS reset counter.
	reset int
	// SQP problem inconsistent state.
	bad bool
	// the initial location.
	x0 []float64 // n
	// the multipliers associated with the general constraints.
	mu []float64 // m
	// the multipliers associated with all constraints (including bounds).
	r []float64 // 𝚖𝚊𝚡(1,m) + n + n
	// the cholesky factor 𝐋𝐃𝐋ᵀ of the approximate hessian 𝐁 of the lagrangian column-wise dense
	// as strict lower triangular 𝐋 with 𝐃 in its diagonal elements.
	l []float64 // ½n×(n+1)+1
	s []float64 // n + 1
	u []float64 // n + 1
	v []float64 // n + 1
	// working space
	w  []float64
	jw []int
	fw findWork
}
