// Copyright ©2025 curioloop. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lbfgsb

import (
	"math"
)

const (
	p5         = 0.5
	p66        = 0.66
	xTrapLower = 1.1
	xTrapUpper = 4.0
)

const (
	stageArmijo = 1
	stageWolfe  = 2
)

type SearchTask int

const (
	SearchStart SearchTask = 0
	SearchConv  SearchTask = 1 << (4 + iota)
	SearchFG
	SearchError
	SearchWarn
)

const (
	SearchErrOverLower = SearchError | (1 + iota)
	SearchErrOverUpper
	SearchErrNegInitG
	SearchErrNegAlpha
	SearchErrNegBeta
	SearchErrNegEps
	SearchErrLower
	SearchErrUpper
	SearchWarnRoundErr = SearchWarn | (1 + iota)
	SearchWarnReachEps
	SearchWarnReachMax
	SearchWarnReachMin
)

type SearchTol struct {
	// Alpha is a non-negative tolerance for the sufficient decrease condition.
	Alpha float64
	// Beta is a non-negative tolerance for the curvature condition.
	Beta float64
	// Eps is a non-negative relative tolerance for an acceptable step.
	// The subroutine exits with a warning if the relative difference between sty and stx is less than Eps.
	Eps float64
	// Lower is a non-negative lower bounds for the step.
	Lower float64
	// Upper is a non-negative upper bounds for the step.
	Upper float64
}

type SearchCtx struct {
	bracket    bool
	stage      int
	g0, gx, gy float64
	f0, fx, fy float64
	stx, sty   float64
	width      [2]float64
	bound      [2]float64
}

// ScalarSearch (dcsrch)
//
// This subroutine finds a step λ that satisfies:
//   - sufficient decrease condition: f(λ) <= f(0) + ɑ*λ*f′(0)
//   - curvature condition: |f'(λ)| <= β*|f'(0)|
//
// Each call of the subroutine updates an interval with endpoints stx and sty.
//
// The interval is initially chosen so that it contains a minimizer of the modified function:
//
//	ψ(λ) = f(λ) - f(0) - ɑ*λ*f′(0)
//
// If ψ(λ) ≤ 0 and f′(λ) ≥ 0 for some step, then the interval is chosen so that it contains a minimizer of f.
//
// If ɑ is less than β and if, for example, the function is bounded below,
// then there is always a step which satisfies both conditions.
//
// If no step can be found that satisfies both conditions, then the algorithm stops with a warning.
// In this case stp only satisfies the sufficient decrease condition.
//
// where
//
//	 f is a double precision variable.
//
//		   On initial entry f is the value of the function at 0.
//		   On subsequent entries f is the value of the function at stp.
//		   On exit f is the value of the function at stp.
//
//	 g is a double precision variable.
//
//		   On initial entry g is the derivative of the function at 0.
//		   On subsequent entries g is the derivative of the function at stp.
//		   On exit g is the derivative of the function at stp.
//
//	 stp is a double precision variable.
//
//		   On entry stp is the current estimate of a satisfactory
//		   step. On initial entry, a positive initial estimate
//		   must be provided.
//		   On exit stp is the current estimate of a satisfactory step
//		   if task = SearchFG. If task = SearchConv then stp satisfies
//		   the sufficient decrease and curvature condition.
func ScalarSearch(f, g, stp float64, task SearchTask, tol *SearchTol, ctx *SearchCtx) (float64, SearchTask) {

	// Initialization block
	if task == SearchStart {
		// Check the input arguments for errors
		if stp < tol.Lower {
			task = SearchErrOverLower
		} else if stp > tol.Upper {
			task = SearchErrOverUpper
		} else if g >= zero {
			task = SearchErrNegInitG
		} else if tol.Alpha < zero {
			task = SearchErrNegAlpha
		} else if tol.Beta < zero {
			task = SearchErrNegBeta
		} else if tol.Eps < zero {
			task = SearchErrNegEps
		} else if tol.Lower < zero {
			task = SearchErrLower
		} else if tol.Upper < tol.Lower {
			task = SearchErrUpper
		}

		// Exit if there are errors on input
		if task&SearchError > 0 {
			return stp, task
		}

		// Initialize local variables
		ctx.bracket = false
		ctx.stage = stageArmijo
		ctx.f0, ctx.g0 = f, g
		ctx.width[0] = tol.Upper - tol.Lower
		ctx.width[1] = ctx.width[0] / p5

		// Initialize the points and their corresponding function and derivative values
		ctx.stx, ctx.fx, ctx.gx = zero, ctx.f0, ctx.g0
		ctx.sty, ctx.fy, ctx.gy = zero, ctx.f0, ctx.g0
		ctx.bound[0] = zero
		ctx.bound[1] = stp + xTrapUpper*(stp)
		task = SearchFG
		return stp, task
	}

	// Test for convergence or warnings
	gTest := tol.Alpha * ctx.g0
	fTest := ctx.f0 + stp*gTest

	stpMin, stpMax := ctx.bound[0], ctx.bound[1]
	if ctx.bracket && (stp <= stpMin || stp >= stpMax) {
		task = SearchWarnRoundErr
	} else if ctx.bracket && (stpMax-stpMin) <= tol.Eps*stpMax {
		task = SearchWarnReachEps
	} else if stp == tol.Upper && f <= fTest && g <= gTest {
		task = SearchWarnReachMax
	} else if stp == tol.Lower && (f > fTest || g >= gTest) {
		task = SearchWarnReachMin
	} else if f <= fTest && math.Abs(g) <= tol.Beta*(-ctx.g0) {
		task = SearchConv
	}

	if task&(SearchWarn|SearchConv) > 0 {
		return stp, task
	}

	if ctx.stage == stageArmijo && f <= fTest && g >= zero {
		ctx.stage = stageWolfe
	}

	if ctx.stage == stageArmijo && f <= ctx.fx && f > fTest {
		fm := f - stp*gTest
		fxm := ctx.fx - ctx.stx*gTest
		fym := ctx.fy - ctx.sty*gTest
		gm := g - gTest
		gxm := ctx.gx - gTest
		gym := ctx.gy - gTest
		scalarStep(&ctx.stx, &fxm, &gxm, &ctx.sty, &fym, &gym, &stp, fm, gm, &ctx.bracket, ctx.bound)
		ctx.fx = fxm + ctx.stx*gTest
		ctx.fy = fym + ctx.sty*gTest
		ctx.gx = gxm + gTest
		ctx.gy = gym + gTest
	} else {
		scalarStep(&ctx.stx, &ctx.fx, &ctx.gx, &ctx.sty, &ctx.fy, &ctx.gy, &stp, f, g, &ctx.bracket, ctx.bound)
	}

	// Decide if a bisection step is needed.
	if ctx.bracket {
		if math.Abs(ctx.sty-ctx.stx) >= p66*ctx.width[1] {
			stp = ctx.stx + p5*(ctx.sty-ctx.stx)
		}
		ctx.width[1] = ctx.width[0]
		ctx.width[0] = math.Abs(ctx.sty - ctx.stx)
	}

	if ctx.bracket {
		stpMin = math.Min(ctx.stx, ctx.sty)
		stpMax = math.Max(ctx.stx, ctx.sty)
	} else {
		stpMin = stp + xTrapLower*(stp-ctx.stx)
		stpMax = stp + xTrapUpper*(stp-ctx.stx)
	}
	ctx.bound[0], ctx.bound[1] = stpMin, stpMax

	stp = math.Min(math.Max(stp, tol.Lower), tol.Upper)

	if ctx.bracket && (stp <= stpMin || stp >= stpMax) || (ctx.bracket && stpMax-stpMin <= tol.Eps*stpMax) {
		stp = ctx.stx
	}

	task = SearchFG
	return stp, task
}

// Subroutine scalarStep (dcstep)
//
// This subroutine computes a safeguarded step for a search
// procedure and updates an interval that contains a step that
// satisfies a sufficient decrease and a curvature condition.
//
// The parameter stx contains the step with the least function
// value. If bracket is set to true then a minimizer has
// been bracketed in an interval with endpoints stx and sty.
// The parameter stp contains the current step.
// The subroutine assumes that if bracket is set to true then
//
//	min(stx,sty) < stp < max(stx,sty),
//
// and that the derivative at stx is negative in the direction
// of the step.
//
// where
//
//	stx is a double precision variable.
//	  On entry stx is the best step obtained so far and is an endpoint of the interval that contains the minimizer.
//	  On exit stx is the updated best step.
//
//	fx is a double precision variable.
//	  On entry fx is the function at stx.
//	  On exit fx is the function at stx.
//
//	dx is a double precision variable.
//	  On entry dx is the derivative of the function at stx.
//	  The derivative must be negative in the direction of the step, that is, dx and stp - stx must have opposite signs.
//
//	  On exit dx is the derivative of the function at stx.
//
//	sty is a double precision variable.
//	  On entry sty is the second endpoint of the interval that contains the minimizer.
//	  On exit sty is the updated endpoint of the interval that contains the minimizer.
//
//	fy is a double precision variable.
//	  On entry fy is the function at sty.
//	  On exit fy is the function at sty.
//
//	dy is a double precision variable.
//	  On entry dy is the derivative of the function at sty.
//	  On exit dy is the derivative of the function at the exit sty.
//
//	stp is a double precision variable.
//	  On entry stp is the current step. If bracket is set to true then on input stp must be between stx and sty.
//	  On exit stp is a new trial step.
//
//	fp is a double precision variable.
//	  On entry fp is the function at stp
//
//	dp is a double precision variable.
//	  On entry dp is the derivative of the function at stp.
//
//	bracket is a logical variable.
//	  On entry bracket specifies if a minimizer has been bracketed.
//	     Initially bracket must be set to .false.
//	  On exit bracket specifies if a minimizer has been bracketed.
//	     When a minimizer is bracketed, bracket is set to true
func scalarStep(
	stx, fx, dx *float64,
	sty, fy, dy *float64,
	stp *float64, fp, dp float64,
	bracket *bool, bound [2]float64) {

	var gamma, p, q, r, s, sgnd, stpc, stpf, stpq, theta, stpmin, stpmax float64

	stpmin, stpmax = bound[0], bound[1]
	sgnd = dp * (*dx / math.Abs(*dx))

	// First case: A higher function value. The minimum is bracketed.
	// If the cubic step is closer to stx than the quadratic step, the cubic step is taken,
	// otherwise the average of the cubic and quadratic steps is taken.
	if fp > *fx {
		theta = three*(*fx-fp)/(*stp-*stx) + *dx + dp
		s = math.Max(math.Max(math.Abs(theta), math.Abs(*dx)), math.Abs(dp))
		gamma = s * math.Sqrt((theta/s)*(theta/s)-(*dx/s)*(dp/s))
		if *stp < *stx {
			gamma = -gamma
		}
		p = (gamma - *dx) + theta
		q = ((gamma - *dx) + gamma) + dp
		r = p / q
		stpc = *stx + r*(*stp-*stx)
		stpq = *stx + ((*dx/((*fx-fp)/(*stp-*stx)+*dx))/two)*(*stp-*stx)
		if math.Abs(stpc-*stx) < math.Abs(stpq-*stx) {
			stpf = stpc
		} else {
			stpf = stpc + (stpq-stpc)/two
		}
		*bracket = true
	} else if sgnd < zero {
		// Second case: A lower function value and derivatives of opposite sign.
		// The minimum is bracketed.
		// If the cubic step is farther from stp than the secant step, the cubic step is taken,
		// otherwise the secant step is taken.
		theta = three*(*fx-fp)/(*stp-*stx) + *dx + dp
		s = math.Max(math.Max(math.Abs(theta), math.Abs(*dx)), math.Abs(dp))
		gamma = s * math.Sqrt((theta/s)*(theta/s)-(*dx/s)*(dp/s))
		if *stp > *stx {
			gamma = -gamma
		}
		p = (gamma - dp) + theta
		q = ((gamma - dp) + gamma) + *dx
		r = p / q
		stpc = *stp + r*(*stx-*stp)
		stpq = *stp + (dp/(dp-*dx))*(*stx-*stp)
		if math.Abs(stpc-*stp) > math.Abs(stpq-*stp) {
			stpf = stpc
		} else {
			stpf = stpq
		}
		*bracket = true
	} else if math.Abs(dp) < math.Abs(*dx) {
		// Third case: A lower function value, derivatives of the same sign,
		// and the magnitude of the derivative decreases.
		// The cubic step is computed only if either:
		//   - the cubic tends to infinity in the direction of the step
		//   - the minimum of the cubic is beyond stp.
		// Otherwise the cubic step is defined to be the secant step.
		theta = three*(*fx-fp)/(*stp-*stx) + *dx + dp
		s = math.Max(math.Max(math.Abs(theta), math.Abs(*dx)), math.Abs(dp))
		// The case gamma = 0 only arises if the cubic does not tend to infinity in the direction of the step.
		gamma = s * math.Sqrt((theta/s)*(theta/s)-(*dx/s)*(dp/s))
		if *stp > *stx {
			gamma = -gamma
		}
		p = (gamma - dp) + theta
		q = (gamma + (*dx - dp)) + gamma
		r = p / q
		if r < zero && gamma != zero {
			stpc = *stp + r*(*stx-*stp)
		} else if *stp > *stx {
			stpc = stpmax
		} else {
			stpc = stpmin
		}
		stpq = *stp + (dp/(dp-*dx))*(*stx-*stp)
		if *bracket {
			// A minimizer has been bracketed.
			// If the cubic step is closer to stp than the secant step, the cubic step is taken,
			// otherwise the secant step is taken.
			if math.Abs(stpc-*stp) < math.Abs(stpq-*stp) {
				stpf = stpc
			} else {
				stpf = stpq
			}
			if *stp > *stx {
				stpf = math.Min(*stp+p66*(*sty-*stp), stpf)
			} else {
				stpf = math.Max(*stp+p66*(*sty-*stp), stpf)
			}
		} else {
			// A minimizer has not been bracketed.
			// If the cubic step is farther from stp than the secant step, the cubic step is taken,
			// otherwise the secant step is taken.
			if math.Abs(stpc-*stp) > math.Abs(stpq-*stp) {
				stpf = stpc
			} else {
				stpf = stpq
			}
			stpf = math.Min(stpmax, stpf)
			stpf = math.Max(stpmin, stpf)
		}
	} else {
		// Fourth case: A lower function value, derivatives of the same sign,
		// and the magnitude of the derivative does not decrease.
		// If the minimum is not bracketed, the step is either stpmin or stpmax,
		// otherwise the cubic step is taken.
		if *bracket {
			theta = three*(fp-*fy)/(*sty-*stp) + *dy + dp
			s = math.Max(math.Max(math.Abs(theta), math.Abs(*dy)), math.Abs(dp))
			gamma = s * math.Sqrt((theta/s)*(theta/s)-(*dy/s)*(dp/s))
			if *stp > *sty {
				gamma = -gamma
			}
			p = (gamma - dp) + theta
			q = ((gamma - dp) + gamma) + *dy
			r = p / q
			stpc = *stp + r*(*sty-*stp)
			stpf = stpc
		} else if *stp > *stx {
			stpf = stpmax
		} else {
			stpf = stpmin
		}
	}

	// Update the interval which contains a minimizer.
	if fp > *fx {
		*sty = *stp
		*fy = fp
		*dy = dp
	} else {
		if sgnd < zero {
			*sty = *stx
			*fy = *fx
			*dy = *dx
		}
		*stx = *stp
		*fx = fp
		*dx = dp
	}

	// Compute the new step.
	*stp = stpf
}
