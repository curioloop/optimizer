// Copyright ©2025 curioloop. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lbfgsb

import (
	"math"
)

const (
	searchNoBnd = 1.0e+10
	searchAlpha = 1.0e-3
	searchBeta  = 0.9
	searchEps   = 0.1
)

const (
	searchBackExit = 20
	searchBackSlow = 10
)

// Perform a line search along dₖ subject to the bounds on the problem.
// The λₖ starts with the unit steplength and ensure fₖ₊₁ = f(xₖ + λₖdₖ), gₖ₊₁ = f′ₖ₊₁ satisfies:
//   - sufficient decrease condition: fₖ₊₁ ≤  ɑλₖgₖᵀdₖ (ɑ = 10⁻³)
//   - curvature condition: |gₖ₊₁ᵀdₖ| ≤ β |gₖᵀdₖ| (β = 0.9)
func performLineSearch(loc *iterLoc, spec *iterSpec, ctx *iterCtx) (info errInfo, done bool) {

	n := spec.n
	x, f, g := loc.x, loc.f, loc.g
	d, t, z := ctx.d, ctx.t, ctx.z

	if n < 0 || n > len(x) || n > len(d) || n > len(t) {
		panic("bound check error")
	}

	ctx.gd = ddot(n, g, 1, d, 1)
	if ctx.numEval == 0 {
		ctx.gdOld = ctx.gd
		if ctx.gd >= zero {
			// Line search is impossible when the directional derivative ≥ 0.
			return errDerivative, false
		}
	}

	ctx.stp, ctx.task = ScalarSearch(f, ctx.gd, ctx.stp, ctx.task, &ctx.searchWork.tol, &ctx.searchWork.ctx)
	done = ctx.task&(SearchConv|SearchWarn|SearchError) > 0

	if !done { // Try another x.
		if ctx.stp == one {
			dcopy(n, z, 1, x, 1) // x = xᶜ
		} else {
			for i := 0; i < n; i++ { // x = λₖdₖ + xₖ
				x[i] = ctx.stp*d[i] + t[i]
			}
		}
	} else if ctx.task&SearchError > 0 {
		info = errLineSearchTol
	}
	return
}

func initLineSearch(loc *iterLoc, spec *iterSpec, ctx *iterCtx) {

	x := loc.x
	d := ctx.d
	b := spec.bounds

	if len(b) > len(d) || len(b) > len(x) {
		panic("bound check error")
	}

	ctx.dSqrt = ddot(spec.n, d, 1, d, 1) // d²
	ctx.dNorm = math.Sqrt(ctx.dSqrt)     // ‖ d ‖₂

	// Determine the maximum step length
	stepMax := searchNoBnd
	if ctx.constrained {
		if ctx.iter == 0 {
			stepMax = one
		} else {
			for i, b := range b {
				if b.hint != bndNo {
					d := d[i]
					if d < zero && b.hint <= bndBoth {
						if span := b.Lower - x[i]; span >= zero {
							stepMax = zero // variable fix at lower bound
						} else if d*stepMax < span {
							stepMax = span / d // constraint search step in bound
						}
					} else if d > zero && b.hint >= bndBoth {
						if span := b.Upper - x[i]; span <= zero {
							stepMax = zero // variable fix at upper bound
						} else if d*stepMax > span {
							stepMax = span / d // constraint search step in bound
						}
					}
				}
			}
		}
	}

	if spec.search == nil {
		spec.search = &SearchTol{
			searchAlpha, searchBeta, searchEps, zero, stepMax}
	}
	ctx.searchWork.tol = *spec.search

	if ctx.iter == 0 && !ctx.boxed {
		ctx.stp = math.Min(one/ctx.dNorm, stepMax)
	} else {
		ctx.stp = one
	}

	ctx.numEval = 0
	ctx.numBack = 0
	ctx.task = SearchStart
}
