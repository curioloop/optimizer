// Copyright ©2025 curioloop. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slsqp

import (
	"math"
)

// LSEI (Least-Squares with linear Equality & Inequality) solves the problem 𝚖𝚒𝚗‖ 𝐄𝐱 - 𝐟 ‖₂ subject to 𝐂𝐱 = 𝐝 and 𝐆𝐱 ≥ 𝐡.
//   - 𝐄 is m × n matrix (no assumption need to be made for its rank)
//   - 𝐱 ∈ ℝⁿ
//   - 𝐟 ∈ ℝᵐ
//   - 𝐂 is m1 × n matrix with 𝚛𝚊𝚗𝚔(𝐂) = k = m1 < n
//   - 𝐝 ∈ ℝᵐ¹
//   - 𝐆 is m2 × n matrix
//   - 𝐡 ∈ ℝᵐ²
//
// # LSE Problem
//
// Consider a LSE (Least-Squares with linear Equality) problem 𝚖𝚒𝚗‖ 𝐀𝐱 - 𝐛 ‖₂ subject to 𝐂𝐱 = 𝐝.
//   - 𝐀 is m × n matrix
//   - 𝐱 ∈ ℝⁿ
//   - 𝐛 ∈ ℝᵐ
//   - 𝐂 is m1 × n matrix with 𝚛𝚊𝚗𝚔(𝐂) = k = m1 < n
//   - 𝐝 ∈ ℝᵐ¹
//
// Given an orthogonal transformation of matrix 𝐂 where 𝐇 and 𝐊 are orthogonal, 𝐑 is full-rank.
//
//	𝐂ₘ₁ₓₙ = 𝐇ₘ₁ₓₘ₁[𝐑ₖₓₖ ೦]𝐊ᵀₘ₁ₓₙ
//
// Its pseudo-inverse is defined by 𝐂⁺ = 𝐊𝐑⁺𝐇ᵀ where  𝐑⁺ = [𝐑⁻¹ ೦].
//
// Define partition 𝐊 = [𝐊₁ 𝐊₂] and [𝐊₁ 𝐊₂]ᵀ𝐱 = [𝐲₁ 𝐲₂]ᵀ where
// 𝐊₁ is an n × k matrix, 𝐊₂ is an n × (n-k) matrix, 𝐲₁ is an k-vector and 𝐲₂ is an (n-k)-vector.
//
// All solutions to the 𝐂𝐱 ≅ 𝐝 are given by 𝐱߮ = [𝐲߫₁ 𝐊₂𝐲₂]ᵀ where
//   - y߫₁ = 𝐊𝐑⁺𝐇ᵀ𝐝 = 𝐂⁺𝐝 is is the unique minimal length solution
//   - 𝐲₂ is arbitrary (n-k)-vector
//
// Since the equality constraints can be represented as a line flat 𝐗 = { 𝐱 : 𝐂𝐱 = 𝐝 },
// the feasible solutions to the origin problem can be treated as 𝐗 = { 𝐱 : 𝐱 = 𝐲߫₁ + 𝐊₂𝐲₂ }.
//
// The problem 𝚖𝚒𝚗‖ 𝐀𝐱 - 𝐛 ‖₂ subject to 𝐱 ∈ 𝐗 is is equivalent to finding a 𝐲₂ that minimizes ‖ 𝐀(𝐲߫₁ + 𝐊₂𝐲₂) - 𝐛 ‖₂.
// Rewrite it to the least-squares problem (𝐀𝐊₂)𝐲₂ ≅ (𝐛-𝐀𝐲߫₁), the unique minimal length solution is given by 𝐲߮₂ = (𝐀𝐊₂)⁺(𝐛-𝐀𝐲߫₁).
// Thus the solution of LSE problem is given by  𝐱߮ = 𝐲߫₁ + 𝐊₂𝐲߮₂ = 𝐂⁺𝐝 + 𝐊₂(𝐀𝐊₂)⁺(𝐛-𝐀𝐲𝐂⁺𝐝)
//
// Assume k = m1 such that 𝐇 = 𝐈 and let 𝐊 satisfied that 𝐂𝐊 is lower triangular matrix, such that
//
//	⎡ 𝐂 ⎤ 𝐊 = ⎡ 𝐂߬₁  ೦  ⎤
//	⎣ 𝐀 ⎦     ⎣ 𝐀߬₁  𝐀߬₂ ⎦
//
// Finally the solution of LSE problem is given by 𝐱߮ = 𝐊[𝐲߮₁ 𝐲߮₂]ᵀ
//   - 𝐲߮₁ is obtained by solving triangular system 𝐂߬₁𝐲₁ = 𝐝
//   - 𝐲߮₂ is obtained by solving least-squares 𝐀߬₂𝐲₂ ≅ 𝐛 - 𝐀߬₁𝐲߮₁
//
// # Reduce to LSI
//
// Using the conclusion of LSE, the equality constraints can be eliminated by introducing
// orthogonal basis 𝐊 = [𝐊₁:𝐊₂] of null space 𝐂𝐊₂ = 0 and let 𝐊ᵀ𝐱 = [𝐲₁ 𝐲₂]ᵀ such that
//
//	             mᶜ  n-mᶜ
//	            ┌┴┐  ┌┴┐
//	⎡ 𝐂 ⎤ 𝐊 = ⎡ 𝐂߬₁   ೦  ⎤ ]╴mᶜ       𝐱 = 𝐊⎡ 𝐲₁ ⎤ ]╴ mᶜ
//	⎥ 𝐄 ⎥     ⎥ 𝐄߬₁   𝐄߬₂ ⎥ ]╴mᵉ            ⎣ 𝐲₂ ⎦ ]╴ n-mᶜ
//	⎣ 𝐆 ⎦     ⎣ 𝐆߬₁   𝐆߬₂ ⎦ ]╴mᵍ
//
// The 𝐲߮₁ is determined as solution of triangular system 𝐂߬₁𝐲₁ = 𝐝,
// and 𝐲߮₂ is the solution of LSI problem 𝚖𝚒𝚗‖ 𝐄߬₂𝐲₂ - (𝐟 - 𝐄߬₂𝐲߮₁) ‖₂ subject to 𝐆߬₂𝐲₂ ≥ 𝐡 - 𝐆߬₁𝐲߮₁.
//
// Finally the solution of LSEI problem is given by 𝐱߮ = 𝐊[𝐲߮₁ 𝐲߮₂]ᵀ.
//
// # Lagrange multiplier
//
// 𝚖𝚒𝚗‖ 𝐄𝐱 - 𝐟 ‖₂ subject to 𝐂𝐱 = 𝐝 and 𝐆𝐱 ≥ 𝐡.
// Substitute LSEI to the KKT conditions:
//   - 𝒇(𝐱) = ½‖ 𝐄𝐱 - 𝐟 ‖₂               →  𝜵𝒇(𝐱) = 𝐄ᵀ(𝐄𝐱 - 𝐟)
//   - 𝒈ⱼ(𝐱) = 𝐝ⱼ - 𝐂ⱼ𝐱 (j = 1 ··· mₑ)   →  𝜵𝒈ⱼ(𝐱) = -𝐂
//   - 𝒈ⱼ(𝐱) = 𝐡ⱼ - 𝐆ⱼ𝐱 (j = mₑ+1 ··· m) →  𝜵𝒈ⱼ(𝐱) = -𝐆
//
// The optimality conditions for LSEI are given:
//   - 𝜵ℒ(𝐱ᵏ,𝛍ᵏ,𝛌ᵏ) = 𝐄ᵀ(𝐄𝐱 - 𝐟) - 𝐂ᵀ𝛍ᵏ - 𝐆ᵀ𝛌ᵏ = 0
//   - 𝛌ᵏⱼ ≥ 0 (j = mₑ+1 ··· m)
//   - 𝛌ᵏⱼ(𝐡ⱼ - 𝐆ⱼ𝐱) = 0 (j = mₑ+1 ··· m)
//
// Multiplier of inequality constraints:
//   - when mᵍ = 0, the ine multiplier 𝛌 = 0
//   - when mᵍ > 0, 𝛌 is solving by LDP
//
// Multiplier of equality constraints is given by 𝛍ᵏ = (𝐂ᵀ)⁻¹[𝐄ᵀ(𝐄𝐱 - 𝐟) - 𝐆ᵀ𝛌ᵏ].
//
// # References
//
//	C.L. Lawson, R.J. Hanson, 'Solving least squares problems' Prentice Hall, 1974. (revised 1995 edition)
//	Chapters 20, Algorithm 20.24.
//	Chapters 23, Section 6.
func LSEI(
	// dim(c) :   formal (lc,n),    actual (mc,n)
	// dim(d) :   formal (lc  ),    actual (mc  )
	c []float64, d []float64,
	// dim(e) :   formal (le,n),    actual (me,n)
	// dim(f) :   formal (le  ),    actual (me  )
	e []float64, f []float64,
	// dim(g) :   formal (lg,n),    actual (mg,n)
	// dim(h) :   formal (lg  ),    actual (mg  )
	g []float64, h []float64,
	lc, mc, le, me, lg, mg, n int,
	// dim(x) :   formal (n   ),    actual (n   )
	x []float64,
	// dim(w) :   2×mc+me+(me+mg)×(n-mc)  for LSEI
	//             + (n-mc+1)×(mg+2)+2×mg  for LSI / HFTI
	w []float64,
	// dim(jw):   max(mg, min(me, n-mc))
	jw []int,
	maxIterLs int,
) (norm float64, mode sqpMode) {

	if n < 1 || mc > n {
		return math.NaN(), BadArgument
	}

	if n > len(x) || mc > len(x) ||
		mc < 0 || mc > len(c) || mc > len(d) ||
		me < 0 || me > len(e) || me > len(f) ||
		mg < 0 || mg > len(g) || mg > len(h) {
		panic("bound check error")
	}

	l := n - mc
	// [mc] reserve for Lagrange multipliers of LSEI equality constraints
	iw := mc
	// [(l+1)×(mg+2)+2×mg] reserve for LSI
	ws := w[iw : iw+(l+1)*(mg+2)+2*mg]
	iw += len(ws)
	// [mc] store Householder pivot for 𝐊
	wp := w[iw : iw+mc]
	iw += len(wp)
	// [me × (n-mc)] store 𝐄߬₂
	we := w[iw : iw+me*l]
	iw += len(we)
	// [me] store (𝐟 - 𝐄߬₂𝐲߮₁)
	wf := w[iw : iw+me]
	iw += len(wf)
	// [mg × (n-mc)] store 𝐆߬₂
	wg := w[iw : iw+mg*l]

	if mc > len(wp) || me > len(wf) {
		panic("bound check error")
	}

	// Triangularize 𝐂 and apply factors to 𝐄 and 𝐆
	for i := 0; i < mc; i++ {
		j := min(i+1, lc-1)
		wp[i] = h1(i, i+1, n, c[i:], lc)
		h2(i, i+1, n, c[i:], lc, wp[i], c[j:], lc, 1, mc-i-1) // 𝐂𝐊 = [𝐂߬₁ ೦]
		h2(i, i+1, n, c[i:], lc, wp[i], e, le, 1, me)         // 𝐄𝐊 = [𝐄߬₁ 𝐄߬₂]
		h2(i, i+1, n, c[i:], lc, wp[i], g, lg, 1, mg)         // 𝐆𝐊 = [𝐆߬₁ 𝐆߬₂]
	}

	// Solve triangular system 𝐂߬₁𝐲₁ = 𝐝
	for i := 0; i < mc; i++ {
		diag := c[i+lc*i]
		if math.Abs(diag) < eps {
			return math.NaN(), LSEISingularC // 𝚛𝚊𝚗𝚔(𝐂) < mc
		}
		x[i] = (d[i] - ddot(i, c[i:], lc, x, 1)) / diag // 𝐲߮₁ = 𝐂߬₁⁻¹𝐝
	}

	// first [mg] of working space store the multiplier return by LDP
	dzero(ws[:mg])

	if mc < n { // 𝚛𝚊𝚗𝚔(𝐂) < n
		for i := 0; i < me; i++ { // 𝐟 - 𝐄߬₂𝐲߮₁
			wf[i] = f[i] - ddot(mc, e[i:], le, x, 1)
		}

		if l > 0 {
			if me > len(we) || mg > len(wg) {
				panic("bound check error")
			}
			for i := 0; i < me; i++ { // 𝐄߬₂
				dcopy(l, e[i+le*mc:], le, we[i:], me)
			}
			for i := 0; i < mg; i++ { // 𝐆߬₂
				dcopy(l, g[i+lg*mc:], lg, wg[i:], mg)
			}
		}

		if mg > 0 {
			for i := 0; i < mg; i++ { // 𝐡 - 𝐆߬₁𝐲߮₁
				h[i] -= ddot(mc, g[i:], lg, x, 1)
			}
			// Compute 𝐲߮₂ by solving 𝚖𝚒𝚗‖ 𝐄߬₂𝐲₂ - (𝐟 - 𝐄߬₂𝐲߮₁) ‖₂  𝚜.𝚝  𝐆߬₂𝐲₂ ≥ 𝐡 - 𝐆߬₁𝐲߮₁.
			norm, mode = LSI(we, wf, wg, h, me, me, mg, mg, l, x[mc:n], ws, jw, maxIterLs)
			if mc == 0 {
				// The multipliers will be return as 𝛌 = w[:mg]
				return
			}
			if mode != HasSolution {
				return math.NaN(), mode
			}
			t := dnrm2(mc, x, 1)
			norm = math.Sqrt(norm*norm + t*t)
		} else {
			k, t := max(le, n), sqrtEps
			var nrm [1]float64
			// Compute 𝐲߮₂ by solving unconstrained 𝚖𝚒𝚗‖ 𝐄߬₂𝐲₂ - (𝐟 - 𝐄߬₂𝐲߮₁) ‖₂
			rank := HFTI(we, me, me, l, wf, k, 1, t, nrm[:], w, w[l:], jw)
			norm = nrm[0]
			dcopy(l, wf, 1, x[mc:n], 1)
			if rank != l {
				return norm, HFTIRankDefect
			}
		}
	}
	for i := 0; i < me; i++ { // 𝐄ᵀ(𝐄𝐱 - 𝐟)
		f[i] = ddot(n, e[i:], le, x, 1) - f[i]
	}
	for i := 0; i < mc; i++ { // 𝐄ᵀ(𝐄𝐱 - 𝐟) - 𝐆ᵀ𝛌
		d[i] = ddot(me, e[i*le:], 1, f, 1) -
			ddot(mg, g[i*lg:], 1, ws[:mg], 1)
	}
	for i := mc - 1; i >= 0; i-- { // 𝐱߮ = 𝐊[𝐲߮₁ 𝐲߮₂]ᵀ
		h2(i, i+1, n, c[i:], lc, wp[i], x, 1, 1, 1)
	}
	for i := mc - 1; i >= 0; i-- { // 𝛍 = (𝐂ᵀ)⁻¹[𝐄ᵀ(𝐄𝐱 - 𝐟) - 𝐆ᵀ𝛌]
		j := min(i+1, lc-1)
		w[i] = (d[i] - ddot(mc-i-1, c[j+lc*i:], 1, w[j:], 1)) / c[i+lc*i]
	}
	// The multipliers will be return as 𝛍 = w[0:mc] and 𝛌 = w[mc:mc+mg]
	mode = HasSolution
	return
}

// LSI (Least-Squares with linear Inequality) solves the problem 𝚖𝚒𝚗‖ 𝐄𝐱 - 𝐟 ‖₂ subject to 𝐆𝐱 ≥ 𝐡.
//   - 𝐄 is m × n matrix with 𝚛𝚊𝚗𝚔(𝐄) = n
//   - 𝐟 ∈ ℝⁿ
//   - 𝐛 ∈ ℝᵐ
//   - 𝐆 is mg × n matrix
//   - 𝐡 ∈ ℝᵐᵍ
//
// Consider below orthogonal decomposition of 𝐄
//
//	                  n    m-n
//	                 ┌┴┐   ┌┴┐
//	𝐄 = 𝐐⎡𝐑 ೦⎤𝐊ᵀ ≡ [ 𝐐₁ : 𝐐₂ ]⎡𝐑⎤ 𝐊ᵀ
//	     ⎣೦ ೦⎦                 ⎣೦⎦
//
// where
//   - 𝐐 is m × m orthogonal
//   - 𝐊 is n × n orthogonal
//   - 𝐑 is n × n non-singular
//
// By introducing orthogonal change of variable 𝐱 = 𝐊ᵀ𝐲 one can obtain
//
//	⎡𝐐₁ᵀ⎤(𝐄𝐱 - 𝐟) = ⎡𝐑𝐲 - 𝐐₁ᵀ𝐟⎤
//	⎣𝐐₂ᵀ⎦          ⎣   𝐐₂ᵀ𝐟  ⎦
//
// Since orthogonal transformation does not change matrix norm and ‖ 𝐐₂ᵀ𝐟 ‖₂ is constant,
// the LSI objective could be rewritten as 𝚖𝚒𝚗‖ 𝐄𝐱 - 𝐟 ‖₂ = 𝚖𝚒𝚗‖ 𝐑𝐲 - 𝐐₁ᵀ𝐟 ‖₂.
//
// By following definitions
//   - 𝐟߫₁ = 𝐐₁ᵀ𝐟
//   - 𝐟߫₂ = 𝐐₂ᵀ𝐟
//   - 𝐳 = 𝐑𝐲 - 𝐟߫₁
//   - 𝐱 = 𝐊𝐑⁻¹(𝐳 + 𝐟߫₁)
//
// the LSI problem is equivalent to LDP problem 𝚖𝚒𝚗 ‖ 𝐳 ‖₂ subject to 𝐆𝐊𝐑⁻¹𝐳 ≥ 𝐡 - 𝐆𝐊𝐑⁻¹𝐟߫₁
// and the residual vector norm of LSI problem can be computed from (‖ 𝐳 ‖₂ + ‖ 𝐟߫₂ ‖₂)¹ᐟ².
//
// # References
//
//	C.L. Lawson, R.J. Hanson, 'Solving least squares problems' Prentice Hall, 1974. (revised 1995 edition)
//	Chapters 23, Section 5.
func LSI(
	// dim(e) :   formal (le,n),    actual (me,n)
	// dim(f) :   formal (le  ),    actual (me  )
	e []float64, f []float64,
	// dim(g) :   formal (lg,n),    actual (mg,n)
	// dim(h) :   formal (lg  ),    actual (mg  )
	g []float64, h []float64,
	le, me, lg, mg, n int,
	// dim(x) :   n
	x []float64,
	// dim(w) :   (n+1)×(mg+2) + 2×mg
	w []float64,
	//  dim(jw):  lg
	jw []int,
	maxIterLs int) (xnorm float64, mode sqpMode) {

	if n < 1 {
		return 0, BadArgument
	}

	// QR-factors of 𝐄 and application to 𝐟.
	for i := 0; i < n; i++ {
		j := min(i+1, n-1)
		t := h1(i, i+1, me, e[i*le:], 1)
		h2(i, i+1, me, e[i*le:], 1, t, e[j*le:], 1, le, n-i-1) // 𝐐𝐄 = 𝐑 (triangular)
		h2(i, i+1, me, e[i*le:], 1, t, f, 1, 1, 1)             // 𝐐𝐟 = [ 𝐟߫₁ : 𝐟߫₂ ]
	}

	// Transform 𝐆 and 𝐡 to get LDP.
	for i := 0; i < mg; i++ {
		for j := 0; j < n; j++ {
			diag := e[j+le*j]
			if math.Abs(diag) < eps || math.IsNaN(diag) {
				return math.NaN(), LSISingularE // 𝚛𝚊𝚗𝚔(𝐄) < n
			}
			// 𝐆𝐊𝐑⁻¹ (𝐊 = 𝐈ₙ)
			g[i+lg*j] = (g[i+lg*j] - ddot(j, g[i:], lg, e[j*le:], 1)) / diag
		}
		h[i] -= ddot(n, g[i:], lg, f, 1) //  𝐡 - 𝐆𝐊𝐑⁻¹𝐟߫₁
	}

	// Solve LDP.
	if xnorm, mode = LDP(mg, n, g, lg, h, x, w, jw, maxIterLs); mode == HasSolution {
		daxpy(n, one, f, 1, x, 1) // 𝐳 + 𝐟߫₁
		for i := n - 1; i >= 0; i-- {
			j := min(i+1, n-1) // 𝐊𝐑⁻¹(𝐳 + 𝐟߫₁)
			x[i] = (x[i] - ddot(n-i-1, e[i+le*j:], le, x[j:], 1)) / e[i+le*i]
		}
		j := min(n, me-1)
		t := dnrm2(me-n, f[j:], 1)           // ‖ 𝐟߫₂ ‖₂
		xnorm = math.Sqrt(xnorm*xnorm + t*t) // (‖ 𝐳 ‖₂ + ‖ 𝐟߫₂ ‖₂)¹ᐟ².
	}
	return
}
