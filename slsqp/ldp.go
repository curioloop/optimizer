// Copyright ©2025 curioloop. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slsqp

import (
	"math"
)

// LDP (Least Distance Programming) solves the problem 𝚖𝚒𝚗 ‖ 𝐱 ‖₂ subject to 𝐆𝐱 ≥ 𝐡.
//   - 𝐆 is m × n matrix (no assumption need to be made for its rank)
//   - 𝐱 ∈ ℝⁿ
//   - 𝐡 ∈ ℝᵐ
//
// NNLS could solve LDP by given:
//   - an (n+1) × m matrix 𝐀 = [𝐆 : 𝐡]ᵀ
//   - an (n+1)-vector 𝐛 = [Oₙ : 1]
//
// Assume m-vector 𝐮 is optimal solution to NNLS solution:
//   - the residual is an (n+1)-vector 𝐫 = 𝐀𝐮 - 𝐛  = [𝐆ᵀ𝐮 : 𝐡ᵀ𝐮 - 1]ᵀ = [𝐫₁ ··· 𝐫ₙ : 𝐫ₙ₊₁]ᵀ
//   - The dual vector is an m-vector 𝐰 = 𝐀ᵀ(𝐛 - 𝐀𝐮) = 𝐀ᵀ𝐫
//
// The 𝐰ᵀ𝐮 = 0 which is given by:
//   - 𝐰ᵢ ≥ 0 → 𝐮ᵢ = 0
//   - 𝐰ᵢ = 0 → 𝐮ᵢ > 0
//
// Thus the norm-2 of NNLS residual satisfied: ‖ 𝐫 ‖₂ = 𝐫ᵀ𝐫 = 𝐫ᵀ(𝐀𝐮 - 𝐛) = (𝐀ᵀ𝐫)𝐮 - 𝐫ᵀ𝐛 = 𝐰ᵀ𝐮 - 𝐫ₙ₊₁ = - 𝐫ₙ₊₁
//   - ‖ 𝐫 ‖₂ > 0 → 𝐫ₙ₊₁ < 0
//   - ‖ 𝐫 ‖₂ = 0 → 𝐫ₙ₊₁ = 0
//
// Constraints 𝐆𝐱 ≥ 𝐡 is satisfied when ‖ 𝐫 ‖₂ > 0 since:
//
//	(𝐆𝐱 - 𝐡)‖ 𝐫 ‖₂ = [𝐆:𝐡][𝐱:-1]ᵀ(-𝐫ₙ₊₁) = 𝐀ᵀ𝐫 = 𝐰 ≥ 0
//
// Substitute LDP to the KKT conditions:
//   - 𝒇(𝐱) = ½‖ 𝐱 ‖₂                   →  𝜵𝒇(𝐱) = 𝐱
//   - 𝒈ⱼ(𝐱) = 0  (j = 1 ··· mₑ)        →  𝜵𝒈ⱼ(𝐱) = 0
//   - 𝒈ⱼ(𝐱) = 𝐡ⱼ -𝐆ⱼ𝐱 (j = mₑ+1 ··· m) →  𝜵𝒈ⱼ(𝐱) = -𝐆
//
// the optimality conditions for LDP are given:
//   - 𝜵ℒ(𝐱ᵏ,𝛌ᵏ) = 𝐱ᵏ - 𝐆ᵀ𝛌ᵏ = 0
//   - 𝛌ᵏⱼ ≥ 0 ∀j
//   - 𝛌ᵏⱼ(𝐡ⱼ -𝐆ⱼ𝐱) = 0 ∀j
//
// Solution of LDP is given by 𝐱 = [𝐫₁ ··· 𝐫ₙ]ᵀ/(-𝐫ₙ₊₁) = 𝐆ᵀ𝐮 / ‖ 𝐫 ‖₂.
// The Lagrange multiplier of LDP inequality constraint 𝛌 = 𝐆⁻¹𝐱 = 𝐮 / ‖ 𝐫 ‖₂.
//
// # References
//
//	C.L. Lawson, R.J. Hanson, 'Solving least squares problems' Prentice Hall, 1974. (revised 1995 edition)
//	Chapters 23, Algorithm 23.27.
func LDP(
	m, n int,
	// 𝐆 : m×n left-side constraint matrix  (there is no restriction on the rank)
	g []float64, mdg int,
	// 𝐡 : m-vector right-side constraint
	h []float64,
	// 𝐱 : an m-vector solution
	x []float64,
	// working space: (n+1)×(m+2)+2m (multiplier of LDP will be store in w[:m] on return)
	w []float64,
	// working space: m
	jw []int,
	maxIter int,
) (xnorm float64, mode sqpMode) {

	if n <= 0 {
		return math.NaN(), BadArgument
	}
	if m <= 0 {
		return 0, OK
	}

	if m > mdg || mdg*n > len(g) || m > len(h) || n > len(x) || (n+1)*(m+2)+2*m > len(w) || m > len(jw) {
		panic("bound check error")
	}

	// 𝐰[:(n+1)×m]                     =  (n+1)×m matrix 𝐀
	// 𝐰[(n+1)×m:(n+1)×(m+1)]          =  (n+1)-vector 𝐛
	// 𝐰[(n+1)×(m+1):(n+1)×(m+2)]      =  (n+1)-vector 𝐳 (working space)
	// 𝐰[(n+1)×(m+2):(n+1)×(m+2)+m]    =  m-vector 𝐮
	// 𝐰[(n+1)×(m+2)+m:(n+1)×(m+2)+2m] =  m-vector 𝐰

	iw := 0
	a := w[iw : iw+m*(n+1)]
	iw += len(a)
	b := w[iw : iw+(n+1)]
	iw += len(b)
	z := w[iw : iw+(n+1)]
	iw += len(z)
	u := w[iw : iw+m]
	iw += len(u)
	dv := w[iw : iw+m]

	for j := 0; j < m; j++ {
		// Copy 𝐆ᵀ into first n rows and m columns of 𝐀
		dcopy(n, g[j:], mdg, a[j*(n+1):], 1)
		// Copy 𝐡ᵀ into row m+1 of 𝐀
		a[j*(n+1)+n] = h[j]
	}

	// Initialize 𝐛
	dzero(b[:n])
	b[n] = one

	var rnorm float64
	rnorm, mode = NNLS(n+1, m, a, n+1, b, u, dv, z, jw, maxIter)

	var fac float64
	if mode == HasSolution {
		if rnorm <= zero { // ‖ 𝐫 ‖₂
			mode = ConsIncompatible
		} else {
			fac = one - ddot(m, h, 1, u, 1) // -𝐫ₙ₊₁ = 1 - 𝐡ᵀ𝐮
			if math.IsNaN(fac) || fac < eps {
				mode = ConsIncompatible
			}
		}
	}
	if mode != HasSolution {
		return math.NaN(), mode
	}

	fac = one / fac
	for j := 0; j < n; j++ { // 𝐆ᵀ𝐮 / ‖ 𝐫 ‖₂
		x[j] = ddot(m, g[mdg*j:], 1, u, 1) * fac
	}

	for j := 0; j < m; j++ { // 𝐮 / ‖ 𝐫 ‖₂
		w[j] = u[j] * fac
	}

	xnorm = dnrm2(n, x, 1)
	return
}
