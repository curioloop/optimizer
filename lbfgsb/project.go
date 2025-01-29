// Copyright ©2025 curioloop. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lbfgsb

import "math"

// Subroutine projGradNorm (projgr)
//
// This subroutine computes the infinity norm of the projected gradient.
func projGradNorm(loc *iterLoc, spec *iterSpec) float64 {

	// next location xₖ₊₁ = xₖ - ɑₖBₖgₖ (ɑₖBₖ > 0)
	// gradient projection P(gᵢ,lᵢ,uᵢ) limit the gradient to feasible region:
	//   𝚙𝚛𝚘𝚓 gᵢ = 𝚖𝚊𝚡(xᵢ - uᵢ, gᵢ) if gᵢ < 0
	//   𝚙𝚛𝚘𝚓 gᵢ = 𝚖𝚒𝚗(xᵢ - lᵢ, gᵢ) if gᵢ > 0
	//   𝚙𝚛𝚘𝚓 gᵢ = gᵢ               otherwise

	n, b, g, x := spec.n, spec.bounds, loc.g, loc.x
	if n < 0 || n > len(b) || n > len(g) || n > len(x) {
		panic("bound check error")
	}

	norm := zero // ‖ 𝚙𝚛𝚘𝚓 g ‖∞
	for i := 0; i < n; i++ {
		b, g := b[i], g[i]
		if b.hint != bndNo {
			if g < zero {
				if b.hint >= bndBoth {
					g = math.Max(x[i]-b.Upper, g)
				}
			} else {
				if b.hint <= bndBoth {
					g = math.Min(x[i]-b.Lower, g)
				}
			}
		}
		norm = math.Max(norm, math.Abs(g))
	}
	return norm
}

// Subroutine projInitActive (active)
//
// This subroutine initializes ctx.where and projects the initial loc.x to the feasible set if necessary.
func projInitActive(loc *iterLoc, spec *iterSpec, ctx *iterCtx) {

	numBnd := 0
	projected, constrained, boxed := false, false, true

	// initial projection P(xᵢ,lᵢ,uᵢ) limit the x to feasible region:
	//   𝚙𝚛𝚘𝚓 xᵢ = uᵢ    if xᵢ > uᵢ
	//   𝚙𝚛𝚘𝚓 xᵢ = lᵢ    if xᵢ < lᵢ
	//   𝚙𝚛𝚘𝚓 xᵢ = xᵢ    otherwise

	n, b, x, where := spec.n, spec.bounds, loc.x, ctx.where
	if n < 0 || n > len(b) || n > len(x) || n > len(where) {
		panic("bound check error")
	}

	for i := 0; i < n; i++ {
		b := b[i]
		if b.hint != bndNo {
			xi := x[i]
			if b.hint <= bndBoth && xi <= b.Lower {
				if xi < b.Lower {
					projected = true
					x[i] = b.Lower
				}
				numBnd++
			} else if b.hint >= bndBoth && xi >= b.Upper {
				if xi > b.Upper {
					projected = true
					x[i] = b.Upper
				}
				numBnd++
			}
		}
	}

	// Initialize ctx.where and assign values to constrained and boxed.
	for i := 0; i < n; i++ {
		b := b[i]
		boxed = boxed && b.hint == bndBoth
		if b.hint == bndNo {
			where[i] = varUnbound
		} else {
			constrained = true
			if b.hint == bndBoth && b.Upper-b.Lower <= zero {
				where[i] = varFixed
			} else {
				where[i] = varFree
			}
		}
	}

	if log := spec.logger; log.enable(LogLast) {
		if projected {
			log.log("The initial X is infeasible. Restart with its projection.\n")
		}
		if !constrained {
			log.log("This problem is unconstrained.\n")
		}
		if log.enable(LogEval) {
			log.log("At X0 %d variables are exactly at the bounds\n", numBnd)
		}
	}

	ctx.projInitX = projected
	ctx.constrained = constrained
	ctx.boxed = boxed
}
