// Copyright ©2025 curioloop. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lbfgsb

import (
	"math"
)

// Subroutine cauchy
//
// Given
//   - xₖ current location
//   - fₖ the function value of f(x)
//   - gₖ the gradient value of f(x)
//   - Sₖ, Yₖ the correction matrices of Bₖ
//
// The quadratic model without bounds of f(x) at xₖ is
//
//	mₖ(x) = fₖ + gₖᵀ(x-xₖ) + ½(x-xₖ)ᵀBₖ(x-xₖ)
//
// This subroutine computes the generalized Cauchy point (GCP), defined as the first local minimizer of mₖ(x),
// along the piece-wise linear path 𝚙𝚛𝚘𝚓(xₖ - tgₖ) obtained by projecting points along the steepest descent direction xₖ - tgₖ onto the feasible region.
//
// Final return
//   - GCP : xᶜ
//   - Cauchy direction : dᶜ = 𝚙𝚛𝚘𝚓(xₖ - tgₖ) - xₖ
func cauchy(loc *iterLoc, spec *iterSpec, ctx *iterCtx) (info errInfo) {

	log := spec.logger

	// Check the status of the variables, reset where(i) if necessary;
	// compute the Cauchy direction d and the breakpoints t; initialize
	// the derivative f1 and the vector p = W'd (for theta = 1).

	// ‖ 𝚙𝚛𝚘𝚓 g ‖∞ = 𝟶  →  ∀ gᵢ = 𝟶
	xcp := ctx.z
	if ctx.sbgNrm <= zero {
		if log.enable(LogLast) {
			log.log("Subgnorm = 0.  GCP = X.\n")
		}
		// xᶜ = x
		dcopy(spec.n, loc.x, 1, xcp, 1)
		return
	}

	m, n := spec.m, spec.n
	theta := ctx.theta
	col, col2 := ctx.col, 2*ctx.col

	// breakpoint t
	//	tᵢ = (xᵢ - uᵢ)/gᵢ  if gᵢ < 𝟶
	//	tᵢ = (xᵢ - lᵢ)/gᵢ  if gᵢ > 𝟶
	//	tᵢ = ∞             otherwise
	t := ctx.t

	// search direction d
	//	dᵢ = 𝟶    if tᵢ = 𝟶
	//	dᵢ = -gᵢ  otherwise
	d := ctx.d

	// corrections of B
	//  W = [ Y θS ]   M = [ -D    Lᵀ  ]⁻¹
	//                     [ L   θSᵀS  ]

	p := ctx.wa[:2*m]      // p = Wᵀd = [Yd θSd]ᵀ
	c := ctx.wa[2*m : 4*m] // c = Wᵀ(xᶜ - x)
	w := ctx.wa[4*m : 6*m] // w = Wᵢ (the row of W corresponding to the breakpoint)
	v := ctx.wa[6*m:]      // v = M? (temporary workspace for subroutine bmv)

	// f′ = gᵀd = -dᵀd = ∑-dᵢ²
	// f″ = -θf′ - pᵀMp
	var f1, f2, orgF2 float64

	nFree := n  // num of free variable
	nBreak := 0 // num of breakpoint

	bkMin, idxMin := zero, 0
	bounded := true

	if log.enable(LogTrace) {
		log.log("---------------- CAUCHY entered-------------------\n")
	}

	// Initialize p to zero and build it up as we determine d.
	if col2 < 0 || col2 > len(p) || col2 > len(c) {
		panic("bound check error")
	}
	for i := 0; i < col2; i++ {
		p[i] = zero
	}

	// In the following loop we determine for each variable its bounds
	// status and its breakpoint, and update p accordingly.
	// Smallest breakpoint is identified.

	x, g := loc.x, loc.g
	b := spec.bounds

	//	order store the breakpoints in the piecewise linear path and free variables encountered
	//	  - order[:left] are indices of breakpoints which have not been encountered;
	//	  - order[left:break] are indices of encountered breakpoints;
	//	  - order[free:n] are indices of variables which have no bounds constraints along the search direction.
	//
	//	where records the status of the current x variables.
	//	  - where[i] = -3 : xᵢ is free and has bounds, but is not moved
	//	  - where[i] =  0 : xᵢ is free and has bounds, and is moved
	//	  - where[i] =  1 : xᵢ is fixed at lᵢ, and uᵢ ≠ lᵢ
	//	  - where[i] =  2 : xᵢ is fixed at uᵢ, and uᵢ ≠ lᵢ
	//	  - where[i] =  3 : xᵢ is always fixed, i.e., uᵢ=xᵢ=lᵢ
	//	  - where[i] = -1 : xᵢ is always free, i.e., it has no bounds.

	where := ctx.where
	order := ctx.index[1]

	if n < 0 || n > len(x) || n > len(g) || n > len(b) ||
		n > len(d) || n > len(xcp) || n > len(where) || n > len(order) {
		panic("bound check error")
	}

	for i := 0; i < n; i++ {
		negG, bnd := -g[i], b[i]
		var tl, tu float64
		if where[i] != varFixed && where[i] != varUnbound {
			// if xᵢ is not a constant and has bounds, compute xᵢ - uᵢ and  xᵢ - lᵢ.
			if bnd.hint <= bndBoth {
				tl = x[i] - bnd.Lower
			}
			if bnd.hint >= bndBoth {
				tu = bnd.Upper - x[i]
			}
			where[i] = varFree
			// If a variable is close enough to a bounds we treat it as at bounds.
			if bnd.hint <= bndBoth && tl <= zero {
				if negG <= zero { // xᵢ ≤ lᵢ and -gᵢ ≤ 𝟶 means xₖ₊₁ᵢ = xₖᵢ - gₖᵢ < lᵢ
					where[i] = varAtLB
				}
			} else if bnd.hint >= bndBoth && tu <= zero {
				if negG >= zero { // xᵢ ≥ uᵢ and -gᵢ ≥ 𝟶 means xₖ₊₁ᵢ = xₖᵢ - gₖᵢ > uᵢ
					where[i] = varAtUB
				}
			} else {
				if math.Abs(negG) <= zero { // gᵢ = 𝟶
					where[i] = varNotMove
				}
			}
		}

		wy := ctx.wy[i*m : (i+1)*m]
		ws := ctx.ws[i*m : (i+1)*m]

		if where[i] != varFree && where[i] != varUnbound {
			d[i] = zero // set dᵢ = 𝟶 for fixed variable
		} else {
			d[i] = negG       // dᵢ = -gᵢ
			f1 -= negG * negG // f' += -dᵢ²
			// pᵢ = -gᵢ * [yᵢ sᵢ]ᵀ
			py, ps := p[:col], p[col:col2]
			if col < 0 || col > len(py) || col > len(ps) {
				panic("bound check error")
			}
			ptr := ctx.head
			for j := 0; j < col; j++ {
				py[j] += wy[ptr] * negG
				ps[j] += ws[ptr] * negG
				ptr = (ptr + 1) % m
			}
			// Handle bounds
			if bnd.hint <= bndBoth && bnd.hint != bndNo && negG < zero {
				// xᵢ + dᵢ is bounded, compute tᵢ
				order[nBreak], t[nBreak] = i, tl/(-negG)
				if nBreak == 0 || t[nBreak] < bkMin {
					bkMin, idxMin = t[nBreak], nBreak
				}
				nBreak++
			} else if bnd.hint >= bndBoth && negG > zero {
				// xᵢ + dᵢ is bounded, compute tᵢ
				order[nBreak], t[nBreak] = i, tu/negG
				if nBreak == 0 || t[nBreak] < bkMin {
					bkMin, idxMin = t[nBreak], nBreak
				}
				nBreak++
			} else {
				// xᵢ + dᵢ is not bounded
				nFree--
				order[nFree] = i
				if math.Abs(negG) > zero { // gᵢ = 𝟶
					bounded = false
				}
			}
		}
	}

	// The smallest breakpoints is t[idxMin]=bkMin.
	// The indices of dᵢ ≠ 𝟶 are now stored in two parts
	// order[0:nBreak]
	// order[nFree:n]

	if theta != one {
		// complete the initialization of p for θ ≠ 1
		ps := p[col:col2]
		dscal(col, theta, ps, 1)
	}

	// Initialize GCP xᶜ = x
	dcopy(n, x, 1, xcp, 1)

	if nBreak == 0 && nFree == n {
		// d is a zero vector, return with the initial xcp as GCP.
		if log.enable(LogVerbose) {
			log.log("Cauchy X =  \n     ")
			for i, x := range xcp {
				log.log("%5.2e ", x)
				if (i+1)%6 == 0 {
					log.log("\n     ")
				}
			}
		}
		return
	}

	// Initialize c = Wᵀ(xᶜ - x) = 𝟶.
	for i := 0; i < col2; i++ {
		c[i] = zero
	}

	// Initialize derivative f'' = -θf'
	f2 = -theta * f1
	orgF2 = f2

	if col > 0 {
		// v = Mp
		if info = bmv(spec, ctx, p, v); info != ok {
			return
		}
		// f″ -= pᵀMp
		f2 -= ddot(col2, v, 1, p, 1)
	}

	// Δt𝚖𝚒𝚗 = -f′/f″
	deltaMin := -f1 / f2
	deltaSum := zero
	ctx.seg = 1

	if log.enable(LogTrace) {
		log.log("There are %v breakpoints\n", nBreak)
	}

	// Find the next smallest breakpoint
	found := nBreak == 0 // goto 888
	nLeft := nBreak
	for iter := 1; nLeft > 0; iter++ {
		var tIdx int
		var tVal, tOld float64
		if iter == 1 {
			// Since we already have the smallest breakpoint we need not do
			// heapsort yet. Often only one breakpoint is used and the
			// cost of heapsort is avoided.
			tVal, tIdx = bkMin, order[idxMin]
		} else {
			if iter == 2 {
				// Replace the already used smallest breakpoint with the
				// breakpoint numbered nBreak > nLast, before heapsort call.
				if nLast := nBreak - 1; idxMin != nLast {
					t[idxMin], t[nLast] = t[nLast], t[idxMin]
					order[idxMin], order[nLast] = order[nLast], order[idxMin]
					// t[idxMin], order[idxMin] = t[nLast], order[nLast]
					// t[nLast], order[nLast] = bkMin, order[idxMin]
				}
			}
			// Update heap structure of breakpoints (if iter=2, initialize heap).
			heapSortOut(nLeft, t, order, iter > 2)
			tOld, tVal, tIdx = t[nLeft], t[nLeft-1], order[nLeft-1]
		}

		// compute dt = t[nLeft] - t[nLeft + 1]
		tDelta := tVal - tOld
		if tDelta != zero && log.enable(LogChange) {
			log.log("Piece    %3d --f1, f2 at start point %.2e %.2e\n", ctx.seg, f1, f2)
			log.log("Distance to the next break point = %.2e\n", tDelta)
			log.log("Distance to the stationary point = %.2e\n", deltaMin)
		}

		// If a minimizer is within this interval, locate the GCP and return.
		if deltaMin < tDelta { // Δt𝚖𝚒𝚗 < Δtᵢ
			found = true
			break // goto 888
		}

		// Otherwise fix one variable and reset the corresponding component of d to zero.
		deltaSum += tDelta
		nLeft--

		if tIdx < 0 || tIdx >= n {
			panic("bound check error")
		}

		dBreak := d[tIdx]          // -gᵢ
		d2Break := dBreak * dBreak // gᵢ²
		d[tIdx] = zero             // dᵢ = 𝟶

		if dBreak > zero {
			xcp[tIdx], where[tIdx] = b[tIdx].Upper, varAtUB // xᶜᵢ = uᵢ (dᵢ > 𝟶)
		} else {
			xcp[tIdx], where[tIdx] = b[tIdx].Lower, varAtLB // xᶜᵢ = lᵢ (dᵢ < 𝟶)
		}
		zBreak := xcp[tIdx] - x[tIdx] // zᵢ = xᶜᵢ - xᵢ

		if log.enable(LogChange) {
			log.log("Variable %v is fixed.\n", tIdx+1)
		}

		if nLeft == 0 && nBreak == n {
			// All n variables are fixed, return with xcp as GCP.
			deltaMin = tDelta
			break // goto 999
		}

		// Update the derivative information.
		ctx.seg++

		// Update f1 and f2.
		// f′ = f′ + f″Δtᵢ + gᵢ² + θgᵢzᵢ - gᵢwᵀᵢMc
		// f″ = f″ - θgᵢ² - 2gᵢwᵀᵢMp -gᵢ²wᵀᵢMwᵢ

		f1 += f2*tDelta + d2Break - theta*dBreak*zBreak
		f2 -= theta * d2Break

		// Process matrix product with middle matrix M
		if col > 0 {
			// c = c + pΔtᵢ
			daxpy(col2, tDelta, p, 1, c, 1)

			// w = Wᵢ (2m)
			w1, w2 := w[:col], w[col:col*2]
			if col > len(w1) || col > len(w2) {
				panic("bound check error")
			}

			wy := ctx.wy[tIdx*m : (tIdx+1)*m] // Yᵢ
			ws := ctx.ws[tIdx*m : (tIdx+1)*m] // Sᵢ
			ptr := ctx.head
			for j := 0; j < col; j++ {
				w1[j] = wy[ptr]
				w2[j] = theta * ws[ptr]
				ptr = (ptr + 1) % m
			}

			// v = Mw (2m)
			if info = bmv(spec, ctx, w, v); info != ok {
				return
			}
			wmc := ddot(col2, c, 1, v, 1) // wMc
			wmp := ddot(col2, p, 1, v, 1) // wMp
			wmw := ddot(col2, w, 1, v, 1) // wMw

			// p = p + g * w
			daxpy(col2, -dBreak, w, 1, p, 1)

			f1 += dBreak * wmc                 // += -gᵢwᵀᵢMc
			f2 += 2.0*dBreak*wmp - d2Break*wmw // += -2gᵢwᵀᵢMp -gᵢ²wᵀᵢMwᵢ
		}

		f2 = math.Max(spec.epsilon*orgF2, f2)
		deltaMin = -f1 / f2 // Δt𝚖𝚒𝚗 = -f′/f″
		if nLeft == 0 && bounded {
			f1, f2, deltaMin = zero, zero, zero
		}
	}

	if nLeft == 0 || found { // Handle not searched variables ...
		if log.enable(LogTrace) {
			log.log("\nGCP found in this segment:\n")
			log.log("Piece    %3d --f1, f2 at start point %.2e %.2e\n", ctx.seg, f1, f2)
			log.log("Distance to the stationary point = %.2e\n", deltaMin)
		}

		deltaMin = math.Max(deltaMin, 0) // Δt𝚖𝚒𝚗 = 𝚖𝚊𝚡(Δt𝚖𝚒𝚗, 𝟶)
		deltaSum += deltaMin             // t𝚘𝚕𝚍 = t𝚘𝚕𝚍 + Δt𝚖𝚒𝚗

		// Move free variables (i.e., the ones w/o breakpoints) and
		// the variables whose breakpoints haven't been reached.
		// xᶜᵢ = xᵢ + t𝚘𝚕𝚍 * dᵢ (dᵢ ≠ 𝟶)
		daxpy(n, deltaSum, d, 1, xcp, 1)
	}

	// Update c = c + Δt𝚖𝚒𝚗 * p = Wᵀ(xᶜ - x)
	// which will be used in computing r = Zᵀ(B(xᶜ - x) + g).
	if col > 0 {
		daxpy(col2, deltaMin, p, 1, c, 1)
	}

	if log.enable(LogVerbose) {
		log.log("Cauchy X =  \n     ")
		for i, x := range xcp {
			log.log("%5.2e ", x)
			if (i+1)%6 == 0 {
				log.log("\n     ")
			}
		}
	}
	if log.enable(LogTrace) {
		log.log("\n---------------- exit CAUCHY ----------------------\n")
	}

	return
}

// Subroutine bmv
//
// Given 2m vector v = [ v₁ v₂ ]ᵀ, calculate matrix product p = Mv with 2m × 2m middle matrix:
//
//			M =［ -D    Lᵀ ]⁻¹
//			    [ L   θSᵀS ]
//
//	 1. Calculate upper triangular matrix Jᵀ by applying Cholesky factorization to
//	    symmetric positive define matrix
//
//	      (θSᵀS+LD⁻¹Lᵀ) = JJᵀ
//
//	 2. Reorder the blocks to get M⁻¹ = (AB)⁻¹ = B⁻¹A⁻¹
//
//	     [ -D    Lᵀ ]  = ［ D¹ᐟ²     O ] [ -D¹ᐟ² D⁻¹ᐟ²Lᵀ ]
//	     [ L   θSᵀS ]     [ -LD⁻¹ᐟ²  J ] [  O    Jᵀ     ]
//
//	 3. Calculate p = Bv by solving B⁻¹p = v
//
//	     [ D¹ᐟ²     O ] [ p₁ ] = [ v₁ ]
//	     [ -LD⁻¹ᐟ²  J ] [ p₂ ]   [ v₂ ]
//
//	 4. Calculate p = ABv = Mv by solving A⁻¹p = Bv
//
//	     [ -D¹ᐟ² D⁻¹ᐟ²Lᵀ ] [ p₁ ] = [ p₁ ]
//	     [  O    Jᵀ     ] [ p₂ ]   [ p₂ ]
func bmv(spec *iterSpec, ctx *iterCtx, v, p []float64) (info errInfo) {

	m := spec.m
	col := ctx.col
	if col == 0 {
		return
	}

	sy := ctx.sy // SᵀY (m × m)
	wt := ctx.wt // JJᵀ (m × m)
	// matrices D and L could calculate from SᵀY
	//   D = 𝚍𝚒𝚊𝚐 { sᵀy }ᵢ₌₁,...,ₙ
	// Lᵢⱼ = { sᵀy₍ᵢⱼ₎ }ᵢ,ⱼ₌ₖ₋ₘ,...,ₖ₋₁ (i > j)

	v1, v2 := v[:col], v[col:2*col]
	p1, p2 := p[:col], p[col:2*col]

	// PART I: Solve  [ D¹ᐟ²     O ] [ p₁ ] = [ v₁ ]
	//                [ -LD⁻¹ᐟ²  J ] [ p₂ ]   [ v₂ ]

	//          D¹ᐟ²p₁ = v₁  ⇒   p₁ = D⁻¹ᐟ²v₁
	// -LD⁻¹ᐟ²p₁ + Jp₂ = v₂  ⇒   p₂ = J⁻¹(v₂ + LD⁻¹v₁)

	// Calculate v₂ + LD⁻¹v₁
	p2[0] = v2[0]
	for i := 1; i < col; i++ {
		// Calculate (LD⁻¹v₁)ᵢ = ∑(Lᵢⱼ * v₁ⱼ / Dⱼⱼ)
		var sum float64
		for j := 0; j < i; j++ {
			sum += sy[i*m+j] * v1[j] / sy[j*m+j]
		}
		// Calculate v₂ᵢ + (LD⁻¹v₁)ᵢ
		p2[i] = v2[i] + sum
	}

	// Calculate p₂ by solving triangular system Jp₂ = v₂ + LD⁻¹v₁
	if dtrsl(wt, m, col, p2, 1, solveUpperT) != 0 {
		return errSingularTriangular
	}

	// Solve p₁ = D⁻¹ᐟ²v₁
	for i := 0; i < col; i++ {
		p1[i] = v1[i] / math.Sqrt(sy[i*m+i])
	}

	// PART II: Solve  [ -D¹ᐟ² D⁻¹ᐟ²Lᵀ ] [ p₁ ] = [ ṗ₁ ]
	//                 [  O    Jᵀ     ] [ p₂ ]   [ ṗ₂ ]

	//               Jᵀp₂ = ṗ₂  ⇒   p₂ = J⁻ᵀṗ₂
	// -D¹ᐟ²p₁ + D⁻¹ᐟ²Lᵀp₂ = ṗ₁  ⇒   p₁ = -D⁻¹ᐟ²(ṗ₁ - D⁻¹ᐟ²Lᵀp₂)

	// Calculate p₂ by solving Jᵀp₂ = ṗ₂
	if dtrsl(wt, m, col, p2, 1, solveUpperN) != 0 {
		return errSingularTriangular
	}

	// Calculate p₁ = -D⁻¹ᐟ²(ṗ₁ - D⁻¹ᐟ²Lᵀp₂)
	//              = -D⁻¹ᐟ²ṗ₁ + D⁻¹Lᵀp₂
	for i := 0; i < col; i++ {
		p1[i] /= -math.Sqrt(sy[i*m+i]) // -D⁻¹ᐟ²ṗ₁
	}
	for i := 0; i < col; i++ {
		// Calculate (D⁻¹Lᵀp₂)ᵢ = ∑(Lⱼᵢ * p₂ⱼ / Dⱼⱼ)
		var sum float64
		for j := i + 1; j < col; j++ {
			sum += sy[j*m+i] * p2[j] / sy[i*m+i]
		}
		// Calculate p₁ᵢ = (D⁻¹ᐟ²ṗ₁)ᵢ + (D⁻¹Lᵀp₂)ᵢ
		p1[i] += sum
	}

	return
}

// Subroutine heapSortOut (hpsolb)
//
// Given t[:n] and order[:n]:
//   - Build min-heap on t[:n] (sorted = false)
//   - Swap the top elements to the tail t[0] ⇄ t[n-1]
//   - Recover heap t[:n-1] by shifting down t[0]
func heapSortOut(n int, t []float64, order []int, sorted bool) {

	if n < 0 || n > len(t) || n > len(order) {
		panic("bound check error")
	}

	if !sorted { // Build heap on t[:n]
		for k := 1; k < n; k++ {
			i := k // Add t[i] to the heap t[:i-1]
			val, idx := t[i], order[i]
			for i > 0 && i < n {
				j := (i - 1) / 2 // Parent of t[i]
				if val < t[j] {  // Shift down the parent
					t[i], order[i] = t[j], order[j]
					i = j
				} else { // Already a heap
					break
				}
			}
			t[i], order[i] = val, idx
		}
	}

	if n > 1 {
		// Pop the least topVal element of heap
		topVal, topIdx := t[0], order[0]
		// Move the bottom element to topVal t[0] = t[n-1] and trim the heap to t[:n-1]
		val, idx := t[n-1], order[n-1]
		// Shifting down the t[0] until heap recover
		i := 0 // t[i] is parent
		for {
			j := 2*i + 1 // Left child
			if j < n {
				// Select the smaller child when right child available
				if j+1 < n && t[j+1] < t[j] {
					j++
				}
				if t[j] < val { // Shift up the smaller child
					t[i], order[i] = t[j], order[j]
					i = j
				} else {
					break // Stop when parent is smaller than children
				}
			} else {
				break
			}
		}
		// Now t[:n-1] is a heap
		t[i], order[i] = val, idx
		// Store the least element val t[n-1]
		t[n-1], order[n-1] = topVal, topIdx
	}
}

// Subroutine freeVar (freev)
//
// This subroutine counts the entering and leaving variables when iter > 0,
// and finds the index set of free and active variables at the GCP.
func freeVar(spec *iterSpec, ctx *iterCtx) bool {

	log := spec.logger

	n := spec.n
	// index[0] gives the free variables based on the determination in cauchy using the array where.
	//	index[:free] are the indices of free variables
	//	index[free:] are the indices of bounds variables
	index := ctx.index[0]
	// index[1] indicates which variables have changed status since the previous iteration.
	//	state[:enter] have changed from bounds to free.
	//	state[leave:] have changed from free to bounds.
	state := ctx.index[1]
	where := ctx.where

	enter, leave := 0, n
	if ctx.iter > 0 && ctx.constrained {
		// Count the entering and leaving variables.
		for _, k := range index[:ctx.free] {
			if where[k] > varFree {
				leave--
				state[leave] = k
				if log.enable(LogChange) {
					log.log("Variable %v leaves the set of free variables\n", k+1)
				}
			}
		}
		for _, k := range index[ctx.free:n] {
			if where[k] <= varFree {
				state[enter] = k
				enter++
				if log.enable(LogChange) {
					log.log("Variable %v enters the set of free variables\n", k+1)
				}
			}
		}
		if log.enable(LogTrace) {
			log.log(" %v variables leave; %v variables enter\n", n-leave, enter)
		}
	}
	ctx.enter = enter
	ctx.leave = leave

	// Find the index set of free and active variables at the GCP.
	free, act := 0, n
	for i := 0; i < n; i++ {
		if where[i] <= varFree {
			index[free] = i
			free++
		} else {
			act--
			index[act] = i
		}
	}
	ctx.free = free
	ctx.active = n - free

	if log.enable(LogTrace) {
		log.log(" %v variables are free at GCP %v \n", ctx.free, ctx.iter+1)
	}

	return (leave < n) || (enter > 0) || ctx.updated
}
