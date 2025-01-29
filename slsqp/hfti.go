// Copyright ©2025 curioloop. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slsqp

import "math"

// HFTI (Householder Forward Triangulation with column Interchanges) solve a least-squares problem linear least squares 𝐀𝐗 ≅ 𝐁.
//   - 𝐀 is m × n matrix with 𝚙𝚜𝚎𝚞𝚍𝚘-𝚛𝚊𝚗𝚔(𝐀) = k
//   - 𝐗 is n × nb matrix having column vectors 𝐱ⱼ
//   - 𝐁 is m × nb matrix
//
// # Basics
//
// Recall the least-squares problem linear least squares 𝐀𝐱 ≅ 𝐛 where 𝚛𝚊𝚗𝚔(𝐀) = k with below orthogonal transformation.
//
//	𝐀ₘₓₙ = 𝐇ₘₓₘ[𝐑ₖₓₖ ೦]𝐊ᵀₙₓₙ   𝐊ᵀ𝐱 = [𝐲₁ 𝐲₂]ᵀ   𝐇ᵀ𝐛 = [𝐠₁ 𝐠₂]ᵀ
//
// where 𝐇 and 𝐊 are orthogonal, 𝐑 is full-rank, 𝐲₁, 𝐠₁ is k-vector and 𝐲₂, 𝐠₂ is (n-k)-vector, such that:
//   - ‖ 𝐀𝐱 - 𝐛 ‖₂ = ‖ 𝐑𝐲₁ - 𝐠₁ ‖₂ + ‖𝐠₂‖₂ (since orthogonal transformation preserve the norm)
//   - 𝚖𝚒𝚗‖ 𝐀𝐱 - 𝐛 ‖₂ = 𝚖𝚒𝚗‖ 𝐑𝐲₁ - 𝐠₁ ‖₂    (since ‖𝐠₂‖₂ is constant)
//   - 𝐲₁ = 𝐑⁻¹𝐠₁                          (since 𝐑 is invertible)
//   - 𝐲₂ is arbitrary                     (usually set 𝐲₂ = O)
//
// The unique solution of minimum length is given by 𝐱 = 𝐊[𝐲₁ 𝐲₂]ᵀ = 𝐊[𝐑⁻¹𝐠₁ ೦]ᵀ and the norm of residual satisfies ‖𝐫‖ = ‖𝐠₂‖.
//
// When 𝚛𝚊𝚗𝚔(𝐀) = k < 𝚖𝚒𝚗(m,n), there exist orthogonal matrix 𝐐 and permutation matrix 𝐏 such that 𝐐𝐀𝐏 = 𝐑
//
//	⎡𝐑₁₁ 𝐑₁₂⎤  where 𝐑₁₁ is k × k matrix, 𝐑₁₂ is k × (n-k) matrix
//	⎣ ೦  𝐑₂₂⎦    and 𝐑₂₂ is (n-k) × (n-k) matrix
//
//	- permutation matrix 𝐏 interchange column of 𝐀 resulting first k columns of 𝐀𝐏 is linearly independent
//	- orthogonal matrix 𝐐 interchange column of 𝐀 resulting 𝐐𝐀𝐏 is zero below the main diagonal
//
// HFTI assume 𝐀 is rank-deficient that make problem very ill-conditioned.
//
// To stabilizing such problem, HFTI first figure out a 𝚙𝚜𝚎𝚞𝚍𝚘-𝚛𝚊𝚗𝚔(𝐀) = k < 𝛍 where 𝛍 = 𝚖𝚒𝚗(m,n) by computing 𝐑.
// By setting 𝐑₂₂ = ೦ and replace the 𝐀 with 𝐀߬ = 𝐐ᵀ[𝐑₁₁ 𝐑₁₂]ₙₓₙ𝐏ᵀ and 𝐛 with 𝐜 = 𝐐ᵀ𝐛 = [𝐜₁ 𝐜₂]ᵀ the problem become 𝐀߬ 𝐱 ≅ 𝐜.
//
// Since [𝐑₁₁:𝐑₁₂]ₖₓₙ is full-row-rank, its triangulation can be obtained by orthogonal transformation 𝐊
// such that [𝐑₁₁:𝐑₁₂]𝐊ₙₓₙ = [𝐖ₖₓₖ:೦] and 𝐊ᵀ𝐱 = [𝐲₁ 𝐲₂]ᵀ.
//   - For forward triangulation, 𝐖 is a non-singular upper triangular matrix
//   - For backward triangulation, 𝐖 is a non-singular lower triangular matrix
//
// The minimum length solution of 𝐀߬ 𝐱 ≅ 𝐜 is given by 𝐱 = 𝐏𝐊[𝐲₁ 𝐲₂]ᵀ = 𝐏𝐊[𝐖⁻¹𝐜₁ ೦]ᵀ.
// Note that 𝐖 is triangular, computation of 𝐖⁻¹𝐜₁ is simple.
//
// # Pseudo Rank
//
// The pseudo-rank is not a nature of 𝐀 but determined by a user-specified tolerance 𝛕 > 0.
// All sub-diagonal elements in 𝐑 = 𝐐𝐀𝐏 are zero and its diagonal elements satisfy |rᵢ₊₁| < |rᵢ| where i = 1, ..., 𝛍-1.
// The pseudo-rank k equal to the number of diagonal elements of 𝐑 exceeding 𝛕 in magnitude.
//
// # Column Pivoting
//
//	𝐏 is constructed as product of 𝛍 transposition matrix 𝐏₁ × ··· × 𝐏ᵤ
//	where 𝐏ⱼ = (j, pⱼ) denotes the interchange between column j and pⱼ.
//
//	𝐐 is constructed as product of 𝛍 Householder matrix 𝐐ᵤ × ··· × 𝐐₁
//	where 𝐐ⱼ corresponding to the j column after interchange interchange.
//
// This column is the best candidate for numerical stability.
// For the construction of j-th Householder transformation, we consider remaining columns j,...,n
// and select the 𝝺-th column whose sum of squares of components in rows j,...,m is greatest.
//
// # Algorithm Outline
//
// HFTI first transforms the augmented matrix [𝐀:𝐁] ≡ [𝐑:𝐂] = [𝐐𝐀𝐏:𝐐𝐁] using
// pre-multiplying Householder transformation 𝐐 with column interchange 𝐏
// where 𝐀𝐏 is linearly independent and 𝐐 resulting all sub-diagonal elements in 𝐀𝐏 are zero.
//
// After determining the pseudo-rank k by diagonal element of 𝐑, apply forward triangulation
// to 𝐑𝐊 = [𝐖:೦] using Householder transformation 𝐊.
//
// Then solve triangular system 𝐖𝐲₁ = 𝐜₁ and apply 𝐊 to 𝐲₁.
// Finally the solution 𝐱 is obtained by rearranging the 𝐊𝐲₁ = 𝐊𝐖⁻¹𝐜₁ by 𝐏.
//
// # Memory Layout
//
// The space of input data 𝐀 is will be modified to store the intermediate results:
//
//	       k        n-k
//	   ┌───┴───┐  ┌──┴──┐
//	⎡ w₁₁ w₁₂ w₁₃ k₁₄ k₁₅ ⎤┐          the data that define 𝐐 occupy the lower triangular part of 𝐀
//	⎥ u₁₂ w₂₂ w₂₃ k₂₄ k₂₅ ⎥├ k        the data that define 𝐊 occupy the rectangular portion of 𝐀
//	⎥ u₁₃ u₂₃ w₃₃ k₃₄ k₃₅ ⎥┘          the data that define 𝐖 occupy the rectangular portion of 𝐀
//	⎥ u₁₄ u₂₄ u₃₄  †   †  ⎥┐
//	⎥ u₁₅ u₂₅ u₃₅ u₄₅  †  ⎥├ n-k
//	⎣ u₁₆ u₂₆ u₃₆ u₄₆ u₅₆ ⎦┘
//
// And 3 × 𝚖𝚒𝚗(m,n) additional working space required:
//
//	g: [ u₁₁ u₂₂ u₃₃ u₄₄ u₅₅ ]    the pivot scalars for 𝐐
//	h: [ k₁₁ k₂₂ k₃₃  †   †  ]    the pivot scalars for 𝐊
//	p: [ p₁  p₂  p₃  p₄  p₅  ]    interchange record define 𝐏
//
// # References
//
//	C.L. Lawson, R.J. Hanson, 'Solving least squares problems' Prentice Hall, 1974. (revised 1995 edition)
//	Chapters 14, Algorithm 14.9.
func HFTI(
	// initially contains the m × n matrix 𝐀, either m ≥ n or m < n is permitted.
	// there is no restriction on 𝚛𝚊𝚗𝚔(𝐀).
	// on return the array will be modified by the subroutine.
	a []float64, mda, m, n int,
	// initially contains the m x nb matrix 𝐁, if nb = 0 the subroutine will make no reference to it.
	// on return the array will contain the n × nb solution 𝐗.
	b []float64, mdb, nb int,
	// absolute tolerance parameter for pseudo-rank determination.
	tau float64,
	// will contain the norm-2 of the residual for the problem defined by the j-th column vector of 𝐁.
	norm []float64,
	// array of working space
	h, g []float64, ip []int) int {

	const factor = 0.001

	diag := min(m, n)
	if diag <= 0 {
		return 0
	}

	if n > len(h) || diag > len(h) || diag > len(ip) {
		panic("bound check error")
	}

	hmax := zero
	for j := 0; j < diag; j++ {
		// Update the squared column lengths and find lmax.
		lmax := j
		if j > 0 {
			v := math.NaN()
			for l := j; l < n; l++ {
				t := a[(j-1)+mda*l]
				if h[l] -= t * t; !(h[l] <= v) {
					lmax, v = l, h[l]
				}
			}
		}
		// Compute squared column lengths and find lmax.
		if j == 0 || factor*h[lmax] < hmax*eps {
			v := math.NaN()
			for l := j; l < n; l++ {
				sm := zero
				for _, t := range a[j+mda*l : m+mda*l] {
					sm += t * t
				}
				if h[l] = sm; !(h[l] <= v) {
					lmax, v = l, h[l]
				}
			}
			hmax = h[lmax]
		}

		// Perform column interchange 𝐏 if needed.
		ip[j] = lmax
		if ip[j] != j {
			c1, c2 := a[mda*j:mda*j+m], a[mda*lmax:mda*lmax+m]
			if m > len(c1) || m > len(c2) {
				panic("bound check error")
			}
			for i := 0; i < m; i++ {
				c1[i], c2[i] = c2[i], c1[i]
			}
			h[lmax] = h[j]
		}

		// Compute the j-th transformation and apply it to 𝐀 and 𝐁.
		i := min(j+1, n-1)
		h[j] = h1(j, j+1, m, a[mda*j:], 1)                          // 𝐐
		h2(j, j+1, m, a[mda*j:], 1, h[j], a[mda*i:], 1, mda, n-j-1) // 𝐑 = 𝐐𝐀𝐏
		h2(j, j+1, m, a[mda*j:], 1, h[j], b, 1, mdb, nb)            // 𝐂 = 𝐐𝐁
	}

	// Determine the pseudo-rank
	// k = 𝚖𝚊𝚡ⱼ |𝐑ⱼⱼ| > 𝛕
	k := diag
	for j := 0; j < diag; j++ {
		if math.Abs(a[j+mda*j]) <= tau {
			k = j
			break
		}
	}

	if k > len(a) || k > len(b) || k > len(g) || nb > len(norm) {
		panic("bound check error")
	}

	// Compute the norms of the residual vectors ‖𝐠₂‖ ≡ ‖𝐜₂‖
	for jb := 0; jb < nb; jb++ {
		sm := zero
		if k < m {
			for _, t := range b[mdb*jb+k : mdb*jb+m] {
				sm += t * t
			}
		}
		norm[jb] = math.Sqrt(sm)
	}

	if k > 0 {
		// If the pseudo-rank is less than n,
		// compute Householder decomposition of first k rows.
		if k < n {
			for i := k - 1; i >= 0; i-- {
				g[i] = h1(i, k, n, a[i:], mda)              // 𝐊
				h2(i, k, n, a[i:], mda, g[i], a, mda, 1, i) // 𝐑₁₁𝐊 = 𝐖
			}
		}

		// If 𝐁 is provided, compute 𝐗
		for jb := 0; jb < nb; jb++ {
			cb := b[mdb*jb:]
			if k > len(cb) || n > len(cb) {
				panic("bound check error")
			}

			// Solve k × k triangular system 𝐖𝐲₁ = 𝐜₁
			for i := k - 1; i >= 0; i-- {
				sm := zero
				for j := uint(i + 1); j < uint(k); j++ {
					sm += a[i+mda*int(j)] * cb[j]
				}
				cb[i] = (cb[i] - sm) / a[i+mda*i]
			}

			// Complete computation of solution vector.
			if k < n {
				dzero(cb[k:n]) // 𝐊𝐲₂ = O
				for i := 0; i < k; i++ {
					h2(i, k, n, a[i:], mda, g[i], cb, 1, mdb, 1) // 𝐊𝐲₁ = 𝐊𝐖⁻¹𝐜₁
				}
			}

			// Re-order solution vector 𝐊𝐲 by 𝐏 to obtain 𝐱.
			for j := diag - 1; j >= 0; j-- {
				if l := ip[j]; ip[j] != j {
					cb[l], cb[j] = cb[j], cb[l]
				}
			}
		}
	} else if nb > 0 {
		for jb := 0; jb < nb; jb++ {
			dzero(b[mdb*jb : mdb*jb+n])
		}
	}

	// The solution vectors 𝐗 are now in the first n rows of 𝐁.
	return k
}
