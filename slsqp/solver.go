// Copyright ©2025 curioloop. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slsqp

import (
	"math"
)

// sqpSolver solve NLP(general constrained NonLinear optimization Problem) with SQP(Sequential Quadratic Programming)
//
// minimize 𝒇(𝐱) subject to
//   - equality constrains: 𝒄ⱼ(𝐱) = 0  (j = 1 ··· mₑ)
//   - inequality constrains: 𝒄ⱼ(𝐱) ≥ 0  (j = mₑ+1 ··· m)
//   - boundaries: 𝒍ᵢ ≤ 𝐱ᵢ ≤ 𝒖ᵢ (i = 1 ··· n)
//
// SQP decomposes NLP into a series of QP sub-problems,
// each of which solves a descent direction 𝐝 and step length 𝛂,
// and ensures that 𝒇(𝐱 + 𝛂𝐝) < 𝒇(𝐱) and the updated 𝐱 satisfies the constraints.
//
// # Direction
//
// The Lagrangian function of NLS is given by ℒ(𝐱,𝛌) = 𝒇(𝐱) - ∑𝛌ⱼ𝒄ⱼ(𝐱)
// which is a linear approximation of constraints 𝒄ⱼ(𝐱).
//
// A quadratic approximation of ℒ(𝐱,𝛌) at location 𝐱ᵏ is a standard form of QP problem:
//
// minimize ½ 𝐝ᵀ𝐁ᵏ𝐝 + 𝜵𝒇(𝐱ᵏ)𝐝 subject to
//   - 𝜵𝒄ⱼ(𝐱ᵏ)𝐝 + 𝒄ⱼ(𝐱ᵏ) = 0  (j = 1 ··· mₑ)
//   - 𝜵𝒄ⱼ(𝐱ᵏ)𝐝 + 𝒄ⱼ(𝐱ᵏ) ≥ 0  (j = mₑ+1 ··· m)
//
// With a symmetric Hessian approximation 𝐁ᵏ ≈ 𝜵²ℒ(𝐱ᵏ,𝛌ᵏ),
// the descent search direction 𝐝 is determined by above problem.
//
// # Inconsistent Constraints
//
// The constraints in QP might become inconsistent with original NLP during the iteration.
// To overcome this difficulty, an augmented QP relaxation with slack variable 𝛅 is introduced to ensure consistency.
//
// minimize ½ 𝐝ᵀ𝐁ᵏ𝐝 + 𝜵𝒇(𝐱ᵏ)𝐝 + ½𝛒𝛅² subject to
//   - 𝜵𝒄ⱼ(𝐱ᵏ)𝐝 + 𝒄ⱼ(𝐱ᵏ) + 𝛅𝒄ⱼ(𝐱ᵏ) = 0  (j = 1 ··· mₑ)
//   - 𝜵𝒄ⱼ(𝐱ᵏ)𝐝 + 𝒄ⱼ(𝐱ᵏ) + 𝛅𝛇ⱼ𝒄ⱼ(𝐱ᵏ) ≥ 0  (j = mₑ+1 ··· m)
//   - 0 ≤ 𝛅 ≤ 1
//   - 𝛇ⱼ = 0 if 𝒄ⱼ(𝐱ᵏ) > 0 (j = mₑ+1 ··· m)
//   - 𝛇ⱼ = 1 if 𝒄ⱼ(𝐱ᵏ) ≤ 0 (j = mₑ+1 ··· m)
//
// where
//   - 10² ≤ 𝛒 ≤ 10⁷ is a constant to penalize the violation of the linear constraints
//   - 𝚭 is an (m-mₑ)×(m-mₑ) diagonal selection matrix with diagonal elements 𝚭ⱼⱼ = 𝛇ⱼ
//
// The augmented direction is given by (n+1)-vector [𝐝 𝛅]ᵀ with initial value [0 ··· 0 1]ᵀ.
// Note that augmented QP is always feasible because 𝐝 = 0 and 𝛅 = 1 can satisfy its constraints trivially.
//
// # Step
//
// The step length 𝛂 is obtained by minimize merit function 𝞿(𝛂) = 𝟇(𝐱 + 𝛂𝐝)
// where 𝟇(𝐱;𝛒) is a non-differentiable function with L1 penalty 𝟇(𝐱;𝛒) = 𝒇(𝐱) + ∑𝛒ⱼ‖𝒄ⱼ(𝐱)‖₁
//   - ‖ 𝒄ⱼ(𝐱) ‖₁ = 𝚊𝚋𝚜[𝒄ⱼ(𝐱)] = 𝚖𝚊𝚡[𝒄ⱼ(𝐱),-𝒄ⱼ(𝐱)]    (j = 1 ··· mₑ)
//   - ‖ 𝒄ⱼ(𝐱) ‖₁ = 𝚊𝚋𝚜[𝚖𝚒𝚗[0,𝒄ⱼ(𝐱)]] = 𝚖𝚊𝚡[0,-𝒄ⱼ(𝐱)] (j = mₑ+1 ··· m)
//
// Maximize the penalty parameters 𝛒 iteratively could lead to optimal solution
//
//	𝛒ⱼᵏ⁺¹ = 𝚖𝚊𝚡[ ½(𝛒ⱼᵏ+|𝛌ⱼ|), |𝛌ⱼ| ] (j = 1 ··· m)
//
// where 𝛍ⱼ is the Lagrange multiplier of j-th constraint.
//
// To overcome possible difficulties in the line search of non-differentiable merit function,
// the update iteration of 𝛒ⱼ is substituted by the differentiable augmented Lagrangian function
//
//	                ⎧ -∑(𝛌ⱼ𝒄ⱼ(𝐱) - ½𝛒ⱼ𝒄ⱼ²(𝐱))  ∀j = 1 ··· mₑ
//	𝞥(𝐱;𝛒) = 𝒇(𝐱) + ⎨ -∑(𝛌ⱼ𝒄ⱼ(𝐱) - ½𝛒ⱼ𝒄ⱼ²(𝐱))  ∀j = 1+mₑ ··· m and 𝒄ⱼ(𝐱) ≤ 𝛌ⱼ/𝛒ⱼ
//	                ⎩ -∑(½𝛌ⱼ²/𝛒ⱼ²)            ∀j = 1+mₑ ··· m and 𝒄ⱼ(𝐱) > 𝛌ⱼ/𝛒ⱼ
//
// then the directional derivative of the merit function along the 𝐝 is given by:
//
//	𝜵𝞥(𝐝;𝐱ᵏ,𝛒ᵏ) = 𝜵𝒇(𝐱ᵏ)ᵀ𝐝 - ∑𝛒ᵏⱼ‖𝒄ⱼ(𝐱ᵏ)‖₁
//
// Finally the step length 𝛂 is obtained by performing line-search along 𝜵𝞥(𝐱,𝛌;𝛒) with Armijio condition:
//
//	𝞥(𝐱ᵏ+𝛂𝐝;𝛌,𝛒) - 𝞥(𝐱ᵏ;𝛌,𝛒) < η · 𝛂 · 𝜵𝞥(𝐝;𝐱ᵏ,𝛒ᵏ) (0<η<0.5)
//
// # Least Squares Sub-Problem
//
// The quasi-newton method BFGS is suitable for it only uses first-order information to approximate the hesse-matrix 𝐁 of Lagrangian function.
// In constrained optimization, 𝐁 > 0 is required to ensure convex. Hence a modified BFGS formula is used:
//   - 𝐁ᵏ⁺¹ = 𝐁ᵏ + 𝐪𝐪ᵀ/𝐪ᵀ𝐬 + 𝐁ᵏ𝐬𝐬ᵀ𝐁ᵏ/𝐬ᵀ𝐁ᵏ𝐬
//   - 𝐬 = 𝐱ᵏ⁺¹ - 𝐱ᵏ
//   - 𝐪 = 𝛉𝛈 + (1-𝛉)𝐁ᵏ𝐬
//   - 𝛈 = 𝜵ℒ(𝐱ᵏ⁺¹,𝛌ᵏ) - 𝜵ℒ(𝐱ᵏ,𝛌ᵏ)
//   - if 𝐬ᵀ𝛈 ≥ ⅕ 𝐬ᵀ𝐁ᵏ𝐬 : 𝛉 = 1
//   - otherwise : 𝛉 = ⅘ 𝐬ᵀ𝐁ᵏ𝐬 / (𝐬ᵀ𝐁ᵏ𝐬 - 𝐬ᵀ𝛈)
//
// In practice, the matrix is presented as 𝐁 = 𝐋𝐃𝐋ᵀ
//   - 𝐋 is a strict lower triangular
//   - 𝐃 is diagonal matrix
//
// By using 𝐋𝐃𝐋ᵀ factorization, the QP sub-problem could be replace by a linear least squares sub-problem:
//
// minimize ‖ 𝐃¹ᐟ²𝐋ᵀ𝐝 + 𝐃⁻¹ᐟ²𝐋⁻¹𝜵𝒇(𝐱ᵏ) ‖₂
//
// subject to
//   - 𝜵𝒄ⱼ(𝐱ᵏ)𝐝 + 𝒄ⱼ(𝐱ᵏ) = 0  (j = 1 ··· mₑ)
//   - 𝜵𝒄ⱼ(𝐱ᵏ)𝐝 + 𝒄ⱼ(𝐱ᵏ) ≥ 0  (j = mₑ+1 ··· m)
//
// The augment QP sub-problem with
//
//	minimize ‖ ⎡ 𝐃¹ᐟ²𝐋ᵀ  O ⎤⎡ 𝐝 ⎤ + ⎡ 𝐃⁻¹ᐟ²𝐋⁻¹𝜵𝒇(𝐱ᵏ)⎤ ‖
//	         ‖ ⎣ O    𝛒¹ᐟ² ⎦⎣ 𝛅 ⎦ + ⎣       O       ⎦ ‖₂
//
// subject to
//   - 𝜵𝒄(𝐱ᵏ)𝐝 + 𝒄(𝐱ᵏ) - 𝛅𝒄(𝐱ᵏ) = 0  (j = 1 ··· mₑ)
//   - 𝜵𝒄(𝐱ᵏ)𝐝 + 𝒄(𝐱ᵏ) - 𝛅𝚭𝒄(𝐱ᵏ) ≥ 0  (j = mₑ+1 ··· m)
//
// # Convergence Criteria
//
// The KKT conditions cannot be satisfied within the required tolerance for many real-world problems due to its scale variance.
//
// SLSQP code in Scipy check convergence by the following three aspects:
//   - feasibility : the summation of violation in all the constraints.
//   - optimality : the decrease potential of the objective function and the weighted constraint infeasibility.
//   - step-length : the 2-norm of the descent direction.
//
// Below criteria are checked after obtaining the solution 𝐝 to the problem QP:
//   - C𝑣𝑖𝑜 = ∑‖𝒄ⱼ(𝐱ᵏ)‖₁
//   - C𝑜𝑝𝑡 = |𝜵𝒇(𝐱ᵏ)ᵀ𝐝| + |𝛌ᵏ|ᵀ×‖𝒄(𝐱ᵏ)‖₁
//   - C𝑠𝑡𝑝 = ‖𝐝‖₂
//
// Below criteria are checked after line-search found the step 𝛂:
//   - Ĉ𝑣𝑖𝑜 = ∑‖𝒄ⱼ(𝐱ᵏ + 𝛂𝐝)‖₁
//   - Ĉ𝑜𝑝𝑡 = |𝒇(𝐱ᵏ + 𝛂𝐝) - 𝒇(𝐱ᵏ)|
//   - Ĉ𝑠𝑡𝑝 = ‖𝐝‖₂
//
// # Reference
//
// Dieter Kraft: "A software package for sequential quadratic programming".
// DFVLR-FB 88-28, 1988
type sqpSolver struct {
	optimizer *Optimizer
	workspace *Workspace
	location  *sqpLoc
}

func (ss *sqpSolver) evalLoc(mode sqpMode) sqpMode {
	o, loc := ss.optimizer, ss.location
	func() {
		defer func() {
			if r := recover(); r != nil {
				mode = BadArgument
			}
		}()
		switch mode {
		case evalFunc:
			loc.f = o.Object(loc.x, nil)
			for j, cons := range o.EqCons {
				loc.c[j] = cons(loc.x, nil)
			}
			for j, cons := range o.NeqCons {
				loc.c[j+o.meq] = cons(loc.x, nil)
			}
		case evalGrad:
			tmp, mda := loc.g[:o.n], max(o.m, 1)
			for i, cons := range o.EqCons {
				cons(loc.x, tmp)
				dcopy(o.n, tmp, 1, loc.a[i:], mda)
			}
			for i, cons := range o.NeqCons {
				cons(loc.x, tmp)
				dcopy(o.n, tmp, 1, loc.a[i+o.meq:], mda)
			}
			o.Object(loc.x, loc.g[:o.n])
		default:
			mode = BadArgument
			return
		}
		mode = OK
	}()
	return mode
}

func (ss *sqpSolver) initCtx() (mode sqpMode) {

	if mode = ss.evalLoc(evalFunc); mode != OK {
		return
	}
	if mode = ss.evalLoc(evalGrad); mode != OK {
		return
	}

	// Initialization for the first iteration
	s, c := &ss.optimizer.sqpSpec, &ss.workspace.sqpCtx
	c.acc = s.Stop.Accuracy
	c.tol = ten * c.acc
	c.iter = 0
	c.reset = 0
	dzero(c.s)
	dzero(c.mu)
	return ss.resetBFGS()
}

func (ss *sqpSolver) resetBFGS() (mode sqpMode) {
	spec, ctx := &ss.optimizer.sqpSpec, &ss.workspace.sqpCtx
	ctx.reset++
	if ctx.reset > 5 {
		// Check relaxed convergence in case of positive directional derivative.
		_, mode = ss.checkConv(ctx.tol, SearchNotDescent)
	} else {
		// 𝐋 = 𝐈 , 𝐃 = 𝐈
		l, n := ctx.l, spec.n
		n2 := (n + 1) * n / 2
		dzero(l[:n2])
		for i, j := 0, 0; i < n; i++ {
			l[j] = one
			j += n - i // diag
		}
	}
	return mode
}

func (ss *sqpSolver) checkConv(tol float64, notConv sqpMode) (h3 float64, mode sqpMode) {
	meq := ss.optimizer.sqpSpec.meq
	for j, c := range ss.location.c {
		h1 := zero
		if j < meq {
			h1 = c
		}
		h3 += math.Max(-c, h1)
	}
	if !ss.checkStop(h3, tol) {
		mode = notConv
	}
	return
}

func (ss *sqpSolver) checkStop(vio, tol float64) bool {
	spec, ctx, loc := &ss.optimizer.sqpSpec, &ss.workspace.sqpCtx, ss.location
	// Ĉ𝑣𝑖𝑜 = ∑‖𝒄ⱼ(𝐱ᵏ + 𝛂𝐝)‖₁
	if vio >= tol || ctx.bad || math.IsNaN(loc.f) {
		return false
	} else {
		stop := spec.Stop
		switch {
		case math.Abs(loc.f-ctx.f0) < tol: // Ĉ𝑜𝑝𝑡 = |𝒇(𝐱ᵏ + 𝛂𝐝) - 𝒇(𝐱ᵏ)|
			return true
		case dnrm2(spec.n, ctx.s, 1) < tol: // Ĉ𝑠𝑡𝑝 = ‖𝐝‖₂
			return true
		case stop.FEvalTolerance >= zero && math.Abs(loc.f) < stop.FEvalTolerance:
			return true
		case stop.FDiffTolerance >= zero && math.Abs(loc.f-ctx.f0) < stop.FDiffTolerance:
			return true
		case stop.XDiffTolerance >= zero:
			n, x, x0, u := spec.n, loc.x, ctx.x0, ctx.u
			dcopy(n, x, 1, u, 1)
			daxpy(n, -1, x0, 1, u, 1)
			return dnrm2(n, u, 1) < stop.XDiffTolerance
		}
		return false
	}
}

func (ss *sqpSolver) updateBFGS() (mode sqpMode) {

	// set loc.g = 𝜵𝒇(𝐱) and loc.a = 𝜵𝒄(𝐱)
	if mode = ss.evalLoc(evalGrad); mode != OK {
		return
	}

	// Update Cholesky-factors of the Hessian matrix by modified BFGS formula
	spec, ctx, loc := &ss.optimizer.sqpSpec, &ss.workspace.sqpCtx, ss.location

	m, n, la := spec.m, spec.n, max(spec.m, 1)
	u, r, v, l, s := ctx.u, ctx.r, ctx.v, ctx.l, ctx.s

	if n < 0 || n > len(v) || n > len(u) {
		panic("bound check error")
	}

	// 𝛈 = 𝜵ℒ(𝐱ᵏ⁺¹,𝛌ᵏ) - 𝜵ℒ(𝐱ᵏ,𝛌ᵏ)
	//   = [𝜵𝒇(𝐱ᵏ⁺¹) - 𝛌𝜵𝒄(𝐱ᵏ⁺¹)] - [𝜵𝒇(𝐱ᵏ) - 𝛌𝜵𝒄(𝐱ᵏ)]
	for i, g := range loc.g[:n] {
		u[i] = g - ddot(m, loc.a[i*la:(i+1)*la], 1, r, 1) - v[i]
	}

	// 𝐋ᵀ𝐬
	for i, k := 0, 0; i < n; i++ {
		k++
		sm := zero
		for _, s := range s[i+1 : n] {
			sm += l[k] * s
			k++
		}
		v[i] = s[i] + sm
	}
	// 𝐃𝐋ᵀ𝐬
	for i, k := 0, 0; i < n; i++ {
		v[i] = l[k] * v[i]
		k += n - i
	}
	// 𝐋𝐃𝐋ᵀ𝐬 = 𝐁ᵏ𝐬
	for i := n - 1; i >= 0; i-- {
		k := i
		sm := zero
		for j, v := range v[:i] {
			sm += l[k] * v
			k += n - 1 - j
		}
		v[i] += sm
	}

	h1 := ddot(n, s, 1, u, 1) // 𝐬ᵀ𝛈
	h2 := ddot(n, s, 1, v, 1) // 𝐬ᵀ𝐁ᵏ𝐬
	h3 := 0.2 * h2
	if h1 < h3 {
		// 𝛉 =  ⅘ 𝐬ᵀ𝐁ᵏ𝐬 / (𝐬ᵀ𝐁ᵏ𝐬 - 𝐬ᵀ𝛈)
		h4 := (h2 - h3) / (h2 - h1)
		h1 = h3
		dscal(n, h4, u, 1)           // 𝛉𝐬ᵀ𝛈
		daxpy(n, one-h4, v, 1, u, 1) // 𝛉𝐬ᵀ𝛈 + 𝐬ᵀ(1-𝛉)𝐁ᵏ𝐬 = 𝐬ᵀ(𝛉𝛈 + (1-𝛉)𝐁ᵏ𝐬) = 𝐬ᵀ𝐪
	}

	if h1 == zero || h2 == zero {
		mode = ss.resetBFGS()
		if ctx.reset > 5 {
			return
		}
	} else {
		// if 𝛉 = 1 : σ𝐳𝐳ᵀ = 𝐬ᵀ𝐪(𝐬ᵀ𝐪)ᵀ / ⅕𝐬ᵀ𝐁ᵏ𝐬
		// otherwise : σ𝐳𝐳ᵀ = 𝛈𝛈ᵀ / 𝐬ᵀ𝛈
		compositeT(uint(n), l, u, +one/h1, nil)
		// σ𝐳𝐳ᵀ = 𝐁ᵏ𝐬(𝐁ᵏ𝐬)ᵀ / 𝐬ᵀ𝐁ᵏ𝐬 = 𝐁ᵏ𝐬𝐬ᵀ𝐁ᵏ / 𝐬ᵀ𝐁ᵏ𝐬
		compositeT(uint(n), l, v, -one/h2, u)
	}

	return
}

// SLSQP (Sequential Least Squares Programming) to solve general nonlinear optimization problems.
func (ss *sqpSolver) mainLoop() (mode sqpMode) {

	loc := ss.location
	ctx := &ss.workspace.sqpCtx
	spec := &ss.optimizer.sqpSpec

	n1 := spec.n + 1
	n2 := spec.n * n1 / 2

	m, meq, n, la := spec.m, spec.meq, spec.n, max(spec.m, 1)
	u, r, v, l, s := ctx.u, ctx.r, ctx.v, ctx.l, ctx.s

	mode = ss.initCtx()
	for mode == OK {

		if ctx.iter++; ctx.iter > spec.Stop.MaxIterations {
			ctx.iter--
			return SQPExceedMaxIter
		}

		// Solve an m×n QP sub-problem to obtained 𝐝 and 𝛌
		// then set ctx.s = 𝐝 and ctx.r = 𝛌

		// Transfer bounds from 𝒍 ≤ 𝐱 ≤ 𝒖 to 𝒍 - 𝐱ᵏ ≤ 𝐝 ≤ 𝒖 - 𝐱ᵏ
		for i, b := range spec.Bounds {
			x := loc.x[i]
			u[i] = b.Lower - x // 𝐱ᵏ + 𝐝 ≥ 𝒍  →  𝐝 ≥ 𝒍 - 𝐱ᵏ
			v[i] = b.Upper - x // 𝐱ᵏ + 𝐝 ≤ 𝒖  →  𝐝 ≤ 𝒖 - 𝐱ᵏ
		}
		_, mode = LSQ(m, meq, n, n2+1,
			l, loc.g, loc.a, loc.c, u, v,
			s, r, ctx.w, ctx.jw, spec.Stop.NNLSIterations, spec.BndInf)

		if mode == LSEISingularC && n == meq {
			mode = ConsIncompatible
		}
		h4 := one
		// If it turns out that the original SQP problem is inconsistent,
		// set ctx.bad = true to prevent termination with convergence on this iteration,
		// even if the augmented problem was solved.
		if ctx.bad = mode == ConsIncompatible; ctx.bad {
			// Form augmented QP relaxation.
			a := loc.a[n*la : n1*la]
			for j, c := range loc.c[:m] {
				if j < meq {
					a[j] = -c // -𝒄ⱼ(𝐱ᵏ)
				} else {
					a[j] = math.Max(-c, zero) // -𝛇ⱼ𝒄ⱼ(𝐱ᵏ)
				}
			}
			loc.g[n] = zero
			l[n2] = hun            // 𝛒 = 10²
			dzero(s[:n])           // 𝐝 = 0
			s[n] = one             // 𝛅 = 1
			u[n], v[n] = zero, one // 0 ≤ 𝛅 ≤ 1

			for relax := 0; relax <= 5; relax++ {
				// Solve m×(n+1) augmented problem
				_, mode = LSQ(m, meq, n1, n2+1, l, loc.g, loc.a, loc.c, u, v,
					s, r, ctx.w, ctx.jw, spec.Stop.NNLSIterations, spec.BndInf)
				h4 = one - s[n] // 1 - 𝛅
				if mode == ConsIncompatible {
					l[n2] *= ten // 𝛒 = 𝛒 × 10
					continue
				}
				break
			}
		}

		// Unable to solve LSQ even the augmented one.
		if mode != HasSolution {
			return
		}

		// Update multipliers for L1-test
		for i, g := range loc.g[:n] {
			// save ctx.r = 𝜵𝒇(𝐱ᵏ) - 𝛌𝜵𝒄(𝐱ᵏ) for BFGS update
			v[i] = g - ddot(m, loc.a[i*la:(i+1)*la], 1, r, 1)
		}

		ctx.f0 = loc.f
		copy(ctx.x0, loc.x)

		gs := ddot(n, loc.g, 1, s, 1) // 𝜵𝒇(𝐱ᵏ)ᵀ𝐝
		h1 := math.Abs(gs)            // C𝑜𝑝𝑡 = |𝜵𝒇(𝐱ᵏ)ᵀ𝐝| + |𝛌ᵏ|ᵀ×‖𝒄(𝐱ᵏ)‖₁
		h2 := zero                    // C𝑣𝑖𝑜 = ∑‖𝒄ⱼ(𝐱ᵏ)‖₁
		for j, c := range loc.c[:m] {
			h3 := zero
			if j < meq {
				h3 = c
			}
			h2 += math.Max(-c, h3)                     // ‖𝒄ⱼ(𝐱ᵏ)‖₁
			h3 = math.Abs(r[j])                        // |𝛌ⱼ|
			h1 += h3 * math.Abs(c)                     // |𝛌ⱼ|×‖𝒄ⱼ(𝐱ᵏ)‖₁
			ctx.mu[j] = math.Max(h3, (ctx.mu[j]+h3)/2) // 𝛒ⱼᵏ⁺¹ = 𝚖𝚊𝚡[ ½(𝛒ⱼᵏ+|𝛌ⱼ|), |𝛌ⱼ| ]
		}

		// Check the convergence criteria for NLP problem,
		// stop if they are satisfied
		if h1 < ctx.acc && h2 < ctx.acc && !ctx.bad && !math.IsNaN(loc.f) {
			return OK
		}

		h1 = zero // ∑𝛒ᵏⱼ‖𝒄ⱼ(𝐱ᵏ)‖₁
		for j, c := range loc.c[:m] {
			h3 := zero
			if j < meq {
				h3 = c
			}
			h1 += ctx.mu[j] * math.Max(-c, h3) // ‖𝒄ⱼ(𝐱ᵏ)‖₁
		}

		// 𝞥(𝐱ᵏ;𝛒) = 𝒇(𝐱ᵏ) + 𝛒ᵏ‖𝒄(𝐱ᵏ)‖₁
		ctx.t0 = loc.f + h1

		// 𝜵𝞥 = 𝜵𝒇(𝐱ᵏ)ᵀ𝐝 - (1 - 𝛅)∑𝛒ᵏⱼ‖𝒄ⱼ(𝐱ᵏ)‖₁
		h3 := gs - h1*h4
		if h3 >= zero {
			// Reset the Hessian matrix when an ascent direction is generated.
			mode = ss.resetBFGS()
			if ctx.reset > 5 {
				return
			}
			continue
		}

		// Conduct the line search with the merit function to get a step length 𝛂,
		// set 𝐱ᵏ⁺¹ = 𝐱ᵏ + 𝛂𝐝 and evaluate 𝒇(𝐱ᵏ⁺¹), 𝒄ⱼ(𝐱ᵏ⁺¹).
		if spec.Line.Exact {
			ctx.line = int(findNoop)
			ss.exactSearch(math.NaN())
		} else {
			ctx.line = 0
			ctx.alpha = spec.Line.Alpha.Upper
			ss.inexactSearch()
			h3 *= ctx.alpha
		}

		for mode = evalFunc; mode == evalFunc; {
			mode = ss.lineSearch(&h3)
		}

		if mode == OK {
			return
		}

		// evaluate 𝜵𝒇(𝐱ᵏ⁺¹), 𝜵𝒄ⱼ(𝐱ᵏ⁺¹) and update BFGS
		if mode == evalGrad {
			mode = ss.updateBFGS()
		}
	}
	return
}

func (ss *sqpSolver) inexactSearch() {
	s, c, x := &ss.optimizer.sqpSpec, &ss.workspace.sqpCtx, ss.location.x
	c.line++
	dscal(s.n, c.alpha, c.s, 1) // 𝐬 = 𝐱ᵏ⁺¹ - 𝐱ᵏ = 𝛂𝐝
	dcopy(s.n, c.x0, 1, x, 1)
	daxpy(s.n, one, c.s, 1, x, 1) // 𝐱ᵏ⁺¹ = 𝐱ᵏ + 𝛂𝐝
	b, inf := s.Bounds, s.BndInf
	for i, v := range x {
		l, u := b[i].Lower, b[i].Upper
		if !math.IsNaN(l) && l > -inf && v < l {
			x[i] = l
		} else if !math.IsNaN(u) && u < inf && v > u {
			x[i] = u
		}
	}
}

func (ss *sqpSolver) exactSearch(t float64) (mode findMode) {
	s, c, x := &ss.optimizer.sqpSpec, &ss.workspace.sqpCtx, ss.location.x
	mode = findMode(c.line)
	if mode != findConv {
		c.alpha, mode = findMin(mode, &c.fw, t, c.tol, *s.Line.Alpha)
		c.line = int(mode)
		dcopy(s.n, c.x0, 1, x, 1)
		daxpy(s.n, c.alpha, c.s, 1, x, 1) // 𝐱 + 𝛂𝐝
	} else {
		dscal(s.n, c.alpha, c.s, 1) // 𝐬 = 𝐱ᵏ⁺¹ - 𝐱ᵏ = 𝛂𝐝
	}
	return
}

func (ss *sqpSolver) lineSearch(h3 *float64) (mode sqpMode) {

	if mode = ss.evalLoc(evalFunc); mode != OK {
		return
	}

	spec, ctx, loc := &ss.optimizer.sqpSpec, &ss.workspace.sqpCtx, ss.location

	// Functions at the current x
	m, meq := spec.m, spec.meq

	// 𝞥(𝐱ᵏ+𝛂𝐝;𝛒) = 𝒇(𝐱ᵏ+𝛂𝐝) + 𝛒ᵏ‖𝒄(𝐱ᵏ+𝛂𝐝)‖₁
	t := loc.f
	for j, c := range loc.c[:m] {
		h1 := zero
		if j < meq {
			h1 = c
		}
		t += ctx.mu[j] * math.Max(-c, h1)
	}

	li := spec.Line
	// Ĉ𝑣𝑖𝑜 = ∑‖𝒄ⱼ(𝐱ᵏ + 𝛂𝐝)‖₁
	// Ĉ𝑜𝑝𝑡 = |𝒇(𝐱ᵏ + 𝛂𝐝) - 𝒇(𝐱ᵏ)|
	if h1 := t - ctx.t0; !li.Exact {
		if h1 <= *h3/10 || ctx.line > 10 {
			*h3, mode = ss.checkConv(ctx.acc, evalGrad)
		} else {
			al, au := li.Alpha.Lower, li.Alpha.Upper
			ctx.alpha = math.Min(math.Max(*h3/(2*(*h3-h1)), al), au)
			// TODO: when alpha is NaN, it can be replaced with alpha.min (but seems not very helpful...)
			ss.inexactSearch()
			*h3 *= ctx.alpha
			mode = evalFunc
		}
	} else {
		if ss.exactSearch(t) == findConv {
			*h3, mode = ss.checkConv(ctx.acc, evalGrad)
		} else {
			mode = evalFunc
		}
	}
	return
}

// LSQ (Least Squares Quadratic programming) solves the problem
//
// minimize ‖ 𝐃¹ᐟ²𝐋ᵀ𝐱 + 𝐃⁻¹ᐟ²𝐋⁻¹𝐠 ‖₂ subject to
//   - 𝐀ⱼ𝐱 - 𝐛ⱼ = 0  (j = 1 ··· mₑ)
//   - 𝐀ⱼ𝐱 - 𝐛ⱼ ≥ 0  (j = mₑ+1 ··· m)
//   - 𝒍ᵢ ≤ 𝐱ᵢ ≤ 𝒖ᵢ (i = 1 ··· n)
//
// where
//   - 𝐋 is an n × n lower triangular matrix with unit diagonal elements
//   - 𝐃 is an n × n diagonal matrix
//   - 𝐠 is an n-vector
//   - 𝐀 is an m × n matrix
//   - 𝐛 is an m-vector
//
// LSQ can be solved as LSEI problem 𝚖𝚒𝚗‖ 𝐄𝐱 - 𝐟 ‖₂ subject to 𝐂𝐱 = 𝐝 and 𝐆𝐱 ≥ 𝐡 with:
//   - 𝐄 = 𝐃¹ᐟ²𝐋ᵀ
//   - 𝐟 = -𝐃⁻¹ᐟ²𝐋⁻¹𝐠
//   - 𝐂 = { 𝐀ⱼ: j = 1 ··· mₑ }
//   - 𝐝 = { -𝐛ⱼ: j = 1 ··· mₑ }
//   - 𝐆ⱼ = { 𝐀ⱼ: j = mₑ+1 ··· m }
//   - 𝐡ⱼ = { -𝐛ⱼ: j = mₑ+1 ··· m }
//
// and the bounds is equivalent to inequality constraints 𝐈𝐱 ≥ 𝒍 and -𝐈𝐱 ≥ -𝒖 such that:
//   - 𝐆ⱼ = { 𝐈ⱼ: j = m+1 ··· m+n }
//   - 𝐡ⱼ = { 𝒍ⱼ: j = m+1 ··· m+n }
//   - 𝐆ⱼ = { -𝐈ⱼ: j = m+n ··· m+2n }
//   - 𝐡ⱼ = { -𝒖ⱼ: j = m+n ··· m+2n }
//
// where
//   - 𝐄 is an n × n upper triangular matrix
//   - 𝐟 is an n-vector
//   - 𝐂 is an mₑ × n matrix
//   - 𝐝 is an mₑ-vector
//   - 𝐆 is an (m-mₑ+2n) × n matrix
//   - 𝐡 is an (m-mₑ+2n)-vector
func LSQ(m, meq, n, nl int,
	// l(nl) = 𝐋 + 𝐃
	// g(n) = 𝐠
	// a(m,n) = 𝐀
	// b(m) = 𝐛
	// xl(n), xu(n) = 𝒍, 𝒖
	l, g, a, b, xl, xu []float64,
	// x(n) : solution vector
	// y(m+n+n) : lagrange multiplier (constraints, lower+upper bounds)
	x, y []float64,
	// w, jw : temporary workspace
	w []float64, jw []int,
	maxIter int, infBnd float64) (float64, sqpMode) {

	mineq := m - meq
	m1 := mineq + n + n // ine
	la := max(m, 1)

	// Determine problem type
	var n1, n2, n3 int
	n1 = n + 1
	if (n+1)*n/2+1 == nl {
		// Solve the origin problem m × n
		n2, n3 = 0, n
	} else {
		// Solve the augmented problem m × (n+1)
		n2, n3 = 1, n-1
	}

	e0, f0 := 0, n*n                // Start index of E and f
	c0, d0 := f0+n, (f0+n)+meq*n    // Start index of C and d
	g0, h0 := d0+meq, (d0+meq)+m1*n // Start index of G and h
	w0 := h0 + m1                   // Start index of workspace

	// Recover matrix E and vector F from l and g
	i2, i3, i4 := 0, 0, 0
	for j := 0; j < n3; j++ {
		i := n - j
		diag := math.Sqrt(l[i2]) // 𝐃¹ᐟ²
		dzero(w[i3 : i3+i])
		dcopy(i-n2, l[i2:], 1, w[i3:], n) // 𝐄ⱼ = 𝐋ⱼᵀ
		dscal(i-n2, diag, w[i3:], n)      //  𝐄ⱼ = 𝐃¹ᐟ²𝐋ⱼᵀ
		w[i3] = diag                      //  𝐄ⱼⱼ = 𝐃¹ᐟ²ⱼⱼ
		// 𝐲 = 𝐋⁻¹𝐠  →  𝐲ⱼ = (𝐠ⱼ - ∑ᵢ𝐋ⱼᵢ𝐲ᵢ) / 𝐋ⱼⱼ
		// 𝐋ⱼⱼ = 1   →  (𝐋⁻¹𝐠)ⱼ = (𝐠ⱼ - ∑ᵢ𝐋ⱼᵢ𝐲ᵢ)
		w[f0+j] = (g[j] - ddot(j, w[i4:], 1, w[f0:], 1)) / diag // 𝐟ⱼ = 𝐃⁻¹ᐟ²ⱼⱼ(𝐋⁻¹𝐠)ⱼ
		i2 += i - n2
		i3 += n1
		i4 += n
	}
	if n2 == 1 {
		w[i3] = l[nl-1]      // 𝐄ⱼⱼ = 𝛒
		dzero(w[i4 : i4+n3]) //
		w[f0+n3] = zero      // 𝐟ⱼ = 0
	}
	dscal(n, -one, w[f0:f0+n], 1) // 𝐟ⱼ = -𝐃⁻¹ᐟ²𝐋⁻¹𝐠

	if meq > 0 {
		// Recover matrix C from upper part of A
		for i := 0; i < meq; i++ {
			dcopy(n, a[i:], la, w[c0+i:], meq) // 𝐂ⱼ = 𝐀ⱼ = - 𝒄ⱼ(𝐱ᵏ)
		}
		// Recover vector d from upper part of b
		dcopy(meq, b, 1, w[d0:], 1) // 𝐝ⱼ = -𝐛ⱼ = -𝒄ⱼ(𝐱ᵏ)
		dscal(meq, -one, w[d0:], 1)
	}

	if mineq > 0 {
		// Recover matrix G from lower part of A
		for i := 0; i < mineq; i++ {
			dcopy(n, a[meq+i:], la, w[g0+i:], m1) // 𝐆ⱼ = 𝐀ⱼ = - 𝒄ⱼ(𝐱ᵏ)
		}
		// Recover vector h from lower part of b
		dcopy(mineq, b[meq:], 1, w[h0:], 1) // 𝐡ⱼ = -𝐛ⱼ = -𝒄ⱼ(𝐱ᵏ)
		dscal(mineq, -one, w[h0:], 1)
	}

	// Augment matrix G with ±𝐈
	// Recover vector h from bounds
	bnd := mineq
	xl, xu = xl[:n], xu[:n]
	for i, l := range xl {
		if !math.IsNaN(l) && l > -infBnd {
			ip, il := g0+bnd, h0+bnd
			w[il] = l    // 𝐡ⱼ = 𝒍ⱼ
			w[ip] = zero // 𝐆ⱼ = 𝐈ⱼ
			dcopy(n, w[ip:], 0, w[ip:], m1)
			w[ip+m1*i] = one
			bnd++
		}
	}
	for i, u := range xu {
		if !math.IsNaN(u) && u < infBnd {
			ip, il := g0+bnd, h0+bnd
			w[il] = -u   // 𝐡ⱼ = -𝒖ⱼ
			w[ip] = zero // 𝐆ⱼ = -𝐈ⱼ
			dcopy(n, w[ip:], 0, w[ip:], m1)
			w[ip+m1*i] = -one
			bnd++
		}
	}

	nan := (n + n) - (bnd - mineq)
	norm, mode := LSEI(w[c0:d0], w[d0:g0], w[e0:f0], w[f0:c0], w[g0:h0], w[h0:w0], max(1, meq), meq, n, n, m1, m1-nan, n, x, w[w0:], jw, maxIter)

	if mode == HasSolution {
		// Restore Lagrange multipliers
		dcopy(m, w[w0:], 1, y, 1)
		if n3 > 0 {
			// Set unused multipliers to NaN
			y[m] = math.NaN()
			dcopy(n3+n3, y[m:], 0, y[m:], 1)
		}
		for i, l := range xl {
			if !math.IsNaN(l) && l > -infBnd && x[i] < l {
				x[i] = l
			}
		}
		for i, u := range xu {
			if !math.IsNaN(u) && u < infBnd && x[i] > u {
				x[i] = u
			}
		}
	}
	return norm, mode
}
