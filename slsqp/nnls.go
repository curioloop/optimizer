// Copyright ©2025 curioloop. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slsqp

import (
	"math"
)

// NNLS (Non-Negative Least-Squares) solve a least-squares problem 𝚖𝚒𝚗 ‖ 𝐀𝐱 - 𝐛 ‖₂ subject to 𝐱 ≥ 0 with active-set method.
//   - 𝐀 is m × n column-major matrix with 𝚛𝚊𝚗𝚔(𝐀) = n (the columns of 𝐀 are linearly independent)
//   - 𝐱 ∈ ℝⁿ
//   - 𝐛 ∈ ℝᵐ
//
// There are two index set ℤ(zero) and ℙ(pivot):
//   - 𝐱ⱼ = 0, j ∈ ℤ : variable indexed in active set ℤ will be held at the value zero
//   - 𝐱ⱼ > 0, j ∈ ℙ : variable indexed in passive set ℙ will be free to take any positive value
//
// When 𝐱ⱼ < 0 occurred, NNLS will change its value to a non-negative value and move its index j from ℙ to ℤ.
//
// The m × k matrix 𝐀ₖ is a subset columns of 𝐀 defined by indices of ℙ.
// NNLS apply QR composition 𝐐𝐀ₖ = [𝐑ₖᵀ:O]ᵀ to solve least-squares [𝐀ₖ:O]𝐱 ≅ 𝐛
// where 𝐐 is m × m orthogonal matrix and 𝐑ₖ is k × k upper triangular matrix.
//
// Once 𝐐 and 𝐑ₖ is computed, the solution is given by 𝐱߮ = [𝐑ₖ⁻¹:O]𝐐𝐛.
//
// Let 𝐛 = [𝐛₁:𝐛₂] (𝐛₁ ∈ ℝⁿ, 𝐛₂ ∈ ℝᵐ⁻ⁿ) and rewrite 𝚖𝚒𝚗‖ 𝐀𝐱 - 𝐛 ‖₂ to 𝚖𝚒𝚗‖ 𝐐ᵀ𝐐[𝐑ₙ:O]𝐱 - 𝐐ᵀ[𝐛₁:𝐛₂] ‖₂
//   - the solution 𝐱 satisfied 𝐑ₙ𝐱 = 𝐐ᵀ𝐛₁ (𝐐ᵀ𝐐 = 𝐈ₘ)
//   - the residual is given by 𝐫 = 𝐐𝐐ᵀ[𝐛₁:𝐛₂]ᵀ - 𝐐[𝐑ₙᵀ𝐱:O]ᵀ = 𝐐[O:𝐐ᵀ𝐛₂]
//   - the norm of residual is given by ‖ 𝐫 ‖₂ = ‖ 𝐐ᵀ𝐛₂ ‖₂
//
// The input will be treated as a whole m × (n+1) working space 𝐐[𝐀:𝐛] where
//   - space of matrix 𝐀 will be used to store the 𝐐𝐀 result
//   - space of vector 𝐛 will be used to store the 𝐐𝐛 result
//
// # Optimality Conditions
//
// Given a problem 𝚖𝚒𝚗 𝒇(𝐱) subject to 𝒉ⱼ(𝐱) = 0 (j = 1 ··· mₑ) and 𝒈ⱼ(𝐱) ≤ 0 (j = mₑ+1 ··· m),
// its optimality at location 𝐱ᵏ are given by below KKT conditions:
//   - 𝜵ℒ(𝐱ᵏ,𝛌ᵏ) = 𝜵𝒇(𝐱ᵏ) + ∑𝛌ᵏⱼ𝜵𝒈ⱼ(𝐱ᵏ) = 0
//   - 𝒈ⱼ(𝐱ᵏ) = 0   (j = 1 ··· mₑ)
//   - 𝒈ⱼ(𝐱ᵏ) ≤ 0   (j = mₑ+1 ··· m)
//   - 𝛌ᵏⱼ ≥ 0      (j = mₑ+1 ··· m)
//   - 𝛌ᵏⱼ𝒈ⱼ(𝐱) = 0  (j = mₑ+1 ··· m)
//
// and substitute NNLS to the KKT conditions:
//   - 𝒇(𝐱) = ½𝐱ᵀ𝐀𝐱 - 2𝐛ᵀ𝐀𝐱 + ½𝐛ᵀ𝐛  →  𝜵𝒇(𝐱) = 𝐀ᵀ(𝐀𝐱 + 𝐛)
//   - 𝒈ⱼ(𝐱) = 0  (j = 1 ··· mₑ)    →  𝜵𝒈ⱼ(𝐱) = 0
//   - 𝒈ⱼ(𝐱) = -𝐱ⱼ (j = mₑ+1 ··· m) →  𝜵𝒈ⱼ(𝐱) = -1
//
// the optimality conditions for NNLS are given:
//   - 𝜵ℒ(𝐱ᵏ,𝛌ᵏ) = 𝐀ᵀ(𝐀𝐱ᵏ + 𝐛) - ∑𝛌ᵏⱼ = 0
//   - 𝛌ᵏⱼ ≥ 0 ∀j
//   - 𝛌ᵏⱼ𝒈ⱼ(𝐱) = 0 ∀j
//
// NNLS introduce a dual m-vector 𝐰 = -𝝺 = -𝜵𝒇(𝐱) = 𝐀ᵀ(𝐛 - 𝐀𝐱) and optimality is given by:
//   - 𝐰ⱼ = 0, ∀j ∈ ℙ
//   - 𝐰ⱼ ≤ 0, ∀j ∈ ℤ
//
// # Active Set Method
//
// The optimality of the activity set method is described by the KKT condition.
//
// Let 𝐱ᵏ be a feasible vector, the inequality constraints 𝒈ⱼ(𝐱ᵏ) (j = mₑ+1 ··· m) has to two status:
//   - active inequality constraints : 𝒈ⱼ(𝐱ᵏ) = 0
//   - passive inequality constraints : 𝒈ⱼ(𝐱ᵏ) < 0
//
// Recall the 𝝺 describes how 𝒇(𝐱) change when relaxing constraints 𝒈ⱼ(𝐱) ≤ 0 → 𝛆 with a interruption 𝛆 > 0:
//   - for 𝛌ⱼ < 0, relax the 𝒈ⱼ(𝐱) will decrease 𝒇(𝐱)
//   - for 𝛌ⱼ > 0, relax the 𝒈ⱼ(𝐱) will increase 𝒇(𝐱)
//
// When we found some active constraints with 𝛌ⱼ < 0:
//   - relax 𝒈ⱼ(𝐱) and move it from ℤ to ℙ
//   - form a new pure equality constrain sub-problem EQP base on new ℤ
//   - solve EQP with variable elimination method
//
// Assume 𝐬 is the EQP solution, then there is 𝒇(𝐬) < 𝒇(𝐱ᵏ) and :
//   - if 𝐬 is feasible, update ℤ and ℙ and solve new EQP until feasible solution is not change
//   - if 𝐬 is infeasible, we just obtain a descending direction 𝐝 = 𝐬 - 𝐱ᵏ and need to find a step length α > 0 such that 𝐱ᵏ + α𝐝 is feasible.
//
// The α can be obtained by projecting the infeasible 𝐬 to the boundaries defined by ℙ.
//
// Once new location 𝐱ᵏ⁺¹ = 𝐱ᵏ + α𝐝 is determined, update ℤ and ℙ and solve new EQP again.
//
// In case of NNLS, the EQP is a unconstrained least-squares problem 𝚖𝚒𝚗 ½‖ 𝐀ᴾ𝐱 - 𝐛 ‖₂.
// The matrix 𝐀ᴾ is a matrix containing only the variables currently in ℙ.
// Thus the solution is given by 𝐬 = [(𝐀ᴾ)ᵀ𝐀ᴾ]⁻¹(𝐀ᴾ)ᵀ𝐛 which is actually computed by QR decomposition.
//
// # Non-negative solution
//
// Consider an m × (n+1) augmented matrix [𝐀:𝐛] defined by least-squares problem 𝐀𝐱 ≅ 𝐛.
//
// Let 𝐐 be an m × m orthogonal matrix that zeros the sub-diagonal elements in first n-1 cols of 𝐀.
//
//	    n     1       n-1  1   1
//	   ┌┴┐   ┌┴┐      ┌┴┐ ┌┴┐ ┌┴┐
//	𝐐[  𝐀 ﹕  𝐛 ] = ⎡  𝐑   𝒔   𝒖 ⎤ ]╴ n-1
//	                ⎣  ０   𝒕   𝒗 ⎦ ]╴ m-n+1
//
// where 𝐑 is an m × m upper triangular full-rank matrix.
//
// Since orthogonal transformation preserves the relationship between the columns of augmented matrix, so there is:
//
//		(𝐐𝐀)ᵀ𝐐𝐛 ＝ 𝐀ᵀ𝐛 ＝ ⎡ 𝑹ᵀ ０ ⎤⎡ 𝒖 ⎤ ＝ ⎡    𝑹ᵀ𝒖   ⎤
//		                  ⎣ 𝒔ᵀ  𝒕ᵀ ⎦⎣ 𝒗 ⎦   ⎣ 𝒔ᵀ𝒖 + 𝒕ᵀ𝒗 ⎦
//
//		                          n-1    1
//		                        ┌──┴──┐ ┌┴┐
//		Assume there is 𝐀ᵀ𝐛 = [ 0 ··· 0  ω ]ᵀ = [𝑹ᵀ𝒖 : 𝒔ᵀ𝒖 + 𝒕ᵀ𝒗]ᵀ.
//	 Since 𝐑 is non-singular, 𝑹ᵀ𝒖 has only the trivial solution 𝒖 = 0 which means 𝒕ᵀ𝒗 = ω.
//
// The n-th component of solution to 𝐀𝐱 ≅ 𝐛 is the least squares solution of 𝒕𝐱ₙ ≅ 𝒗 which is 𝐱ₙ = 𝒕ᵀ𝒗/𝒕ᵀ𝒕 = ω/𝒕ᵀ𝒕.
//
// Thus when the n-th component of 𝐀ᵀ𝐛 is positive (ω > 0), then the n-th component of solution satisfied 𝐱ₙ > 0.
//
// # References
//
//	C.L. Lawson, R.J. Hanson, 'Solving least squares problems' Prentice Hall, 1974. (revised 1995 edition)
//	Chapters 23, Algorithm 23.10.
func NNLS(
	m, n int,
	// initially contains the m × n matrix 𝐀, either m ≥ n or m < n is permitted.
	// there is no restriction on 𝚛𝚊𝚗𝚔(𝐀).
	// on return the array will contain the product matrix 𝐐𝐀 generated implicitly by this routine.
	a []float64, mda int,
	// initially contains the m-vector 𝐛.
	// on return the array will contain the product matrix 𝐐𝐛 generated implicitly by this routine.
	b []float64,
	// will contain the solution vector 𝐱 of primal problem.
	x []float64,
	// will contain the dual vector 𝐰 describe the weight of constraint.
	w []float64,
	// array of working space
	z []float64, index []int,
	// maximum number of iterations
	maxIter int) (float64, sqpMode) {

	const factor = 0.01

	if m <= 0 || n <= 0 || mda < m ||
		len(a) < mda*n || len(b) < m || len(x) < n || len(w) < n || len(z) < m || len(index) < n {
		return math.NaN(), BadArgument
	}

	if maxIter <= 0 {
		maxIter = 3 * n
	}

	np := 0 // num of elem in set ℙ
	z1 := 0 // start index of set ℤ

	// index = ℙ ∪ ℤ = {1,···,n}
	// ℙ = index[:np] define the subset columns of 𝐀
	// ℤ = index[z1:]
	index = index[:n]
	for i := range index {
		index[i] = i
	}

	// Start from 𝐱 = O and all indices are initially in ℤ.
	dzero(x[:n])

	// Calculate norm-2 of the residual vector when return.
	iter := 0
	term := func() (rnorm float64, mode sqpMode) {
		if np < m { // m > 𝚛𝚊𝚗𝚔(𝐀)
			rnorm = dnrm2(m-np, b[np:], 1) // ‖ 𝐐ᵀ𝐛₂ ‖₂
		} else {
			dzero(w[:n])
		}
		if iter > maxIter {
			mode = NNLSExceedMaxIter
		} else {
			mode = HasSolution
		}
		return
	}

	// The main loop is continued until no more active constraints can be set free.
	for {
		if z1 >= n || // Quit if all coefficients are positive : ℤ = ∅ (𝐱 ≥ 0),
			np >= m { // or if m columns of 𝐀 have been triangularized.
			return term()
		}

		// Compute components of the dual vector 𝐰 = 𝐀ᵀ(𝐛 - 𝐀𝐱) (negative gradient).
		// Since 𝐰ⱼ = 0 for j ∈ ℙ, thus we only compute the 𝐰ⱼ for j ∈ ℤ.
		// GIven 𝐱ⱼ = 0 for j ∈ ℤ , the update finally simplified to 𝐰 = 𝐀ᵀ𝐛.
		for _, j := range index[z1:] {
			w[j] = ddot(m-np, a[np+mda*j:], 1, b[np:], 1)
		}

		for {
			// Find index t ∈ ℤ such that 𝐰ₜ = 𝚊𝚛𝚐 𝚖𝚊𝚡 { 𝐰ⱼ: j ∈ ℤ }
			wmax, izmax := zero, 0
			for i, j := range index[z1:] {
				if w[j] > wmax {
					wmax, izmax = w[j], z1+i
				}
			}

			// Quit when 𝐰ⱼ ≤ 0, ∀j ∈ ℤ (no more constraint could be relaxed)
			// this indicates satisfaction of the Kuhn-Tucker conditions
			if wmax <= zero {
				return term()
			}

			// Move index t from ℤ to ℙ
			iz := izmax
			j := index[iz]
			aj := a[mda*j : mda*j+m : mda*j+m]

			// Given j-th column of 𝐀, compute corresponding Householder vector 𝐮.
			asave := aj[np]              // Save the pivot-th component of j-th column 𝐀ₚⱼ.
			up := h1(np, np+1, m, aj, 1) // Now the pivot-th component of j-th column is (𝐐𝐀)ₚⱼ.
			// The pivot-th component of 𝐮 is return as 𝐮ₚ.

			// Check new diagonal element to avoid near linear dependence.
			accept := false
			unorm := dnrm2(np, aj, 1) // ‖ 𝐮 ‖₂
			if math.Abs(aj[np])*factor >= unorm*eps {
				// If column j is sufficiently independent.
				// Compute Householder transformation z = 𝐐𝐛 = [ -σ‖𝐛‖₂ 0 ··· 0 ]ᵀ
				copy(z[:m], b[:m])
				h2(np, np+1, m, aj, 1, up, z, 1, 1, 1)
				// Solve 𝐐(𝐀𝐱)ⱼ ≅ 𝐐𝐛ⱼ for proposed new value for 𝐱ⱼ
				ztest := z[np] / aj[np] // 𝐱 = (𝐐𝐀)⁺𝐐𝐛
				accept = ztest > zero   // 𝐱ⱼ > 0
			}

			if !accept {
				// Reject j as a candidate to be moved from ℤ to ℙ,
				// restore 𝐀ₚⱼ and test dual coefficients again.
				aj[np] = asave
				w[j] = zero
				continue
			}

			// Now the index j=index(iz) is selected.

			// Update b = 𝐐𝐛.
			copy(b[:m], z[:m])

			// Move j from ℤ to ℙ.
			index[iz] = index[z1]
			index[z1] = j
			z1++
			np++

			// Apply Householder transformations to cols in new ℤ.
			if z1 < n {
				for _, jj := range index[z1:] {
					h2(np-1, np, m, aj, 1, up, a[jj*mda:], 1, mda, 1)
				}
			}
			// Zero sub-diagonal elements in col j.
			if np < m {
				dzero(aj[np:m])
			}
			// Set 𝐰ⱼ = 0 for j ∈ ℙ
			w[j] = zero
			break
		}

		// When new j join in the ℙ, the coefficients of the free variables in the unconstrained solution 𝐬 my turn negative.
		// The inner loop is continued until all violating variables have been moved to ℤ.
		for {
			// Compute EQP solution 𝐬 by solving triangular system 𝐱߮ = [𝐑ₖ⁻¹:O]𝐐𝐛
			for ip, jj := np-1, -1; ip >= 0; ip-- {
				if jj >= 0 {
					daxpy(ip+1, -z[ip+1], a[jj*mda:], 1, z, 1)
				}
				jj = index[ip]
				z[ip] /= a[ip+jj*mda]
			}

			// Check iteration count
			if iter++; iter > maxIter {
				return term()
			}

			// See if all new constrained coefficients are feasible.

			// Find index t ∈ ℙ such that 𝐱ₜ/(𝐱ₜ-𝐳ₜ) = 𝚊𝚛𝚐 𝚖𝚒𝚗 { 𝐱ⱼ/(𝐱ⱼ-𝐳ⱼ) : 𝐳ⱼ ≤ 0, j ∈ ℙ }
			alpha, jj := two, -1
			for ip, l := range index[:np] {
				if z[ip] <= zero { // found unfeasible coefficients
					// if not compute alpha.
					t := -x[l] / (z[ip] - x[l])
					if alpha > t { // ɑ = 𝐱ₜ/(𝐱ₜ-𝐳ₜ)
						alpha, jj = t, ip
					}
				}
			}

			// If all coefficients are feasible, exit secondary loop to main loop.
			if jj < 0 {
				for ip, idx := range index[:np] {
					x[idx] = z[ip]
				}
				break // goto mainLoop
			}

			// Interpolate between x and z
			// 𝐱 = 𝐱 + ɑ(𝐬 - 𝐱)
			for ip, l := range index[:np] {
				x[l] += alpha * (z[ip] - x[l])
			}

			// Move coefficient i from ℙ to ℤ
			i := index[jj]
			for {
				x[i] = zero
				if jj++; jj < np {
					for j := jj; j < np; j++ {
						ii := index[j]
						ci := a[ii*mda:]
						index[j-1] = ii
						var cc, ss float64
						cc, ss, ci[j-1] = g1(ci[j-1], ci[j])
						ci[j] = zero
						for l := 0; l < n; l++ {
							if l != ii {
								cl := a[l*mda : l*mda+j+1 : l*mda+j+1]
								cl[j-1], cl[j] = g2(cc, ss, cl[j-1], cl[j])
							}
						}
						b[j-1], b[j] = g2(cc, ss, b[j-1], b[j])
					}
				}

				np--
				z1--
				index[z1] = i

				// See if the remaining coefficients in ℙ are feasible.
				// They should be because of the way ɑ was determined.
				// If any are infeasible, it is due to round-off error.
				// Any that are non-positive will be set to zero and moved from ℙ to ℤ.
				for _, idx := range index[:np] {
					if x[idx] <= zero {
						continue
					}
				}
				break
			}

			// copy b into z.
			// then solve again and loop back.
			copy(z[:m], b[:m])
		}
	}
}
