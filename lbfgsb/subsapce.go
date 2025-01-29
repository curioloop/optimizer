// Copyright ©2025 curioloop. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lbfgsb

import "math"

const (
	solutionUnknown   = -1
	solutionWithinBox = 0
	solutionBeyondBox = 1
)

// Subroutine optimalDirection (subsm)
//
// This subroutine computes an approximate solution of the subspace problem
//
//	  m߮ₖ(d߮) ≡ d߮ᵀr߮ᶜ + ½d߮ᵀB߮ₖr߮ᶜ
//
//	along the subspace unconstrained Newton direction
//
//	  d߮ᵘ = -B߮ₖ⁻¹r߮ᶜ
//
//	then backtrack towards the feasible region to obtain optimal direction (optional)
//
//	  d߮⁎= ɑ⁎ × d߮ᵘ
//
//	Given the L-BFGS matrix and the Sherman-Morrison formula
//
//	  B߮ₖ = (1/θ)I - (1/θ)ZᵀW[ (I-(1/θ)MWᵀZZᵀW)⁻¹M ]WᵀZ(1/θ)
//
//	With N ≡ I - (1/θ)MWᵀZZᵀW, the formula for the unconstrained Newton direction is
//
//	  d߮ᵘ = (1/θ)r߮ᶜ + (1/θ²)ZᵀWN⁻¹MZᵀW
//
//	Then form middle K = M⁻¹N = (N⁻¹M)⁻¹ to avoid inverting N (see formk)
//
//	  d߮ᵘ = (1/θ)r߮ᶜ + (1/θ²)ZᵀWK⁻¹WᵀZr߮ᶜ
//
//	Finally the computation of K⁻¹v could be replaced with solving v = Kx by factorization K = LELᵀ
func optimalDirection(loc *iterLoc, spec *iterSpec, ctx *iterCtx) (info errInfo) {

	free := ctx.free
	if free <= 0 {
		return
	}

	n, m, b := spec.n, spec.m, spec.bounds
	col, head := ctx.col, ctx.head

	m2 := 2 * m
	col2 := 2 * col

	inx := ctx.index[0]
	theta := ctx.theta

	// On exit Cauchy point xᶜ (ctx.z) become the subspace minimizer x̂ over the subspace of free variables.
	x := ctx.z
	// On exit reduced gradient r߮ᶜ (ctx.r) become the Newton direction d߮ᵘ
	d := ctx.r
	// xp is used to safeguard the projected Newton direction
	xp := ctx.xp

	// corrections of B
	//  W = [ Y θS ]   M = [ -D    Lᵀ  ]⁻¹
	//                     [ L   θSᵀS  ]
	ws, wy := ctx.ws, ctx.wy // S, Y

	// M⁻¹N = K = LELᵀ = [  LLᵀ          L⁻¹(-Laᵀ+Rzᵀ)]
	//                   [(-La +Rz)L⁻ᵀ   S'AA'Sθ      ]
	wn := ctx.wn

	wv := ctx.wa[:m2] // v = K⁻¹WᵀZr߮ᶜ (temporary workspace)

	if n < 0 || n > len(x) || n > len(xp) || col < 0 || col > len(wv) ||
		free > len(d) || free > len(x) || free > len(b) || free > len(inx) {
		panic("bound check error")
	}

	log := spec.logger
	if log.enable(LogTrace) {
		log.log("----------------SUBSM entered-----------------\n")
	}

	// Compute v = WᵀZr߮ᶜ
	ptr := head
	for i := 0; i < col; i++ {
		yr := zero
		sr := zero
		for j, k := range inx[:free] { // Free variables in Z
			yr += wy[k*m+ptr] * d[j]
			sr += ws[k*m+ptr] * d[j]
		}
		wv[i] = yr
		wv[col+i] = theta * sr
		ptr = (ptr + 1) % m
	}

	// Compute K⁻¹v = (LELᵀ)⁻¹v = (L⁻ᵀE⁻¹L⁻¹)v
	// Lᵀ stored in the upper triangle of WN
	// E⁻¹ = [-I  0]⁻¹ = [-I  0]
	//       [ 0  I]     [ 0  I]

	// Compute L⁻¹v by solving Lx = (Lᵀ)ᵀx= v
	if dtrsl(wn, m2, col2, wv, 1, solveUpperT) != 0 {
		return errSingularTriangular
	}
	// Compute E⁻¹(L⁻¹v)
	dscal(col, -one, wv, 1)
	// Compute L⁻ᵀ(E⁻¹L⁻¹v) by solving Lᵀx = E⁻¹L⁻¹v
	if dtrsl(wn, m2, col2, wv, 1, solveUpperN) != 0 {
		return errSingularTriangular
	}

	// Compute r߮ᶜ + (1/θ)ZᵀW(K⁻¹WᵀZr߮ᶜ)
	ptr = head
	for jy := 0; jy < col; jy++ {
		js := col + jy
		for i, k := range inx[:free] { // Free variables in Z
			d[i] += (wy[k*m+ptr] * wv[jy] / theta) + (ws[k*m+ptr] * wv[js])
		}
		ptr = (ptr + 1) % m
	}

	// Scale r߮ᶜ + (1/θ)ZᵀWK⁻¹WᵀZr߮ᶜ by 1/θ
	for i := 0; i < free; i++ {
		d[i] *= one / theta
	}

	// Let us try the projection
	dcopy(n, x, 1, xp, 1)

	// Perform projection along unconstrained Newton direction d߮ᵘ
	// Compute subspace minimizer x̂ = 𝚙𝚛𝚘𝚓(xᶜ + d߮ᵘ)
	projected := false
	for i, k := range inx[:free] {
		dk := d[i]
		xk := x[k]
		l, u := b[k].Lower, b[k].Upper
		switch b[k].hint {
		case bndNo:
			x[k] = xk + dk // unbound variable
		case bndLow:
			x[k] = math.Max(l, xk+dk)
			projected = projected || x[k] == l
		case bndUp:
			x[k] = math.Min(u, xk+dk)
			projected = projected || x[k] == u
		case bndBoth:
			x[k] = math.Min(u, math.Max(l, xk+dk))
			projected = projected || x[k] == l || x[k] == u
		}
	}

	if projected {
		ctx.word = solutionBeyondBox
	} else {
		ctx.word = solutionWithinBox
		// return
	}

	// Check sign of the directional derivative
	sgn := zero
	if projected {
		xx, gg := loc.x, loc.g // xₖ, gₖ
		if n < 0 || n > len(x) || n > len(xx) || n > len(gg) {
			panic("bound check error")
		}
		for i := 0; i < n; i++ {
			sgn += (x[i] - xx[i]) * gg[i] // (x̂ - xₖ) * gₖ
		}
	}

	// sgn ≤ 0  ⇒  d߮⁎ = d߮ᵘ
	// sgn > 0  ⇒  d߮⁎ = ɑ⁎ × d߮ᵘ

	// When the direction x̂ - xₖ is not a direction of strong descent for the objective function,
	// truncating the path from xₖ to x̂ to satisfy the constraints
	if sgn > zero {

		copy(x[:n], xp[:n])

		if log.enable(LogLast) {
			log.log("Positive dir derivative in projection.\n")
			log.log("Using the backtracking step.\n")
		}

		// search positive optimal step
		// ɑ⁎ = 𝚖𝚊𝚡 { ɑ : ɑ ≤ 1, lᵢ - xᶜᵢ ≤ ɑ × d߮ᵘᵢ ≤ uᵢ - xᶜᵢ (i ∈ 𝓕) }
		alpha := one

		stp := alpha
		ibd := 0
		for i, k := range inx[:free] {
			dk := d[i]
			bk := b[k]
			if bk.hint != bndNo {
				if dk < zero && bk.hint <= bndBoth {
					if span := bk.Lower - x[k]; span >= 0 {
						stp = zero
					} else if dk*alpha < span {
						stp = span / dk
					}
				} else if dk > zero && bk.hint >= bndBoth {
					if span := bk.Upper - x[k]; span <= 0 {
						stp = zero
					} else if dk*alpha > span {
						stp = span / dk
					}
				}
				if stp < alpha {
					alpha = stp
					ibd = i
				}
			}
		}

		if alpha < one {
			dk := d[ibd]
			k := inx[ibd]
			if dk > zero {
				x[k] = b[k].Upper
				d[ibd] = zero
			} else if dk < zero {
				x[k] = b[k].Lower
				d[ibd] = zero
			}
		}

		// x̂ = xᶜ + d߮⁎ = xᶜ + (ɑ⁎ × d߮ᵘ)
		//   x̂ᵢ = xᶜᵢ         if i ∉ 𝓕
		//   x̂ᵢ = xᶜᵢ + Zd߮⁎ᵢ  otherwise
		for i, k := range inx[:free] {
			x[k] += alpha * d[i]
		}
	}

	if log.enable(LogTrace) {
		log.log("----------------exit SUBSM --------------------\n")
	}

	return
}

// Subroutine reduceGradient (cmprlb)
//
// This subroutine computes r߮ᶜ = -Zᵀ(g + B(xᶜ - xₖ))
func reduceGradient(loc *iterLoc, spec *iterSpec, ctx *iterCtx) (info errInfo) {

	x, g := loc.x, loc.g
	n, m := spec.n, spec.m

	theta := ctx.theta
	index := ctx.index[0]
	col, head, free := ctx.col, ctx.head, ctx.free

	// corrections of B
	//  W = [ Y θS ]   M = [ -D    Lᵀ  ]⁻¹
	//                     [ L   θSᵀS  ]

	z := ctx.z // xᶜ
	r := ctx.r // r = -rᶜ = -Zᵀ(g + θ(xᶜ-x) - WMc) = Zᵀ(-g - θ(xᶜ-x) + WMc)

	c := ctx.wa[2*m : 4*m] // c = Wᵀ(xᶜ - x)
	v := ctx.wa[:2*m]      // v = Mc (temporary workspace for subroutine bmv)

	if (n < 0 || n > len(r) || n > len(g)) || (col < 0 || col > len(v)) ||
		(free < 0 || free > len(r)) || (len(z) != len(x) || len(z) != len(g)) {
		panic("bound check error")
	}

	if !ctx.constrained && col > 0 {
		// If the problem is unconstrained and `col > 0`, set r = -g
		for i := 0; i < n; i++ {
			r[i] = -g[i]
		}
	} else {
		// Compute r = -θ(xᶜ-x) - g for free variables
		for i, k := range index[:free] {
			r[i] = -theta*(z[k]-x[k]) - g[k]
		}

		// Compute v = Mc
		if info = bmv(spec, ctx, c, v); info != ok {
			return
		}

		// Compute r += WMc
		ptr := head
		ws, wy := ctx.ws, ctx.wy
		for j := 0; j < col; j++ {
			mc1, mc2 := v[j], theta*v[col+j]
			for i, k := range index[:free] {
				r[i] += wy[k*m+ptr]*mc1 + ws[k*m+ptr]*mc2 // [ Y θS ]ᵀ[ Mc₁ Mc₂ ]
			}
			ptr = (ptr + 1) % m
		}
	}

	return
}
