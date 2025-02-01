// Copyright ©2025 curioloop. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slsqp

import (
	"math"
)

var sqrtEps = math.Sqrt(eps)              // square root of machine precision
var invPhi2 = one / (math.Phi * math.Phi) //  golden section ratio

// Given m-vector v, h1 construct m×m Householder vector u and scalar s for transformation Qv ≡ y.
// The Householder matrix could be computed with Q = Iₘ - b⁻¹uuᵀ where b = suₚ.
//
// lₚ is the index of the pivot element, which should satisfy 0 ≤ lₚ < l₁.
// If l₁ < m, the transformation will be constructed to zero out elements indexed from l₁ through m.
// But if l₁ ≥ m, the subroutine does an identity transformation.
//
// On input, v contains the pivot vector, ive is the storage increment between elements.
// On output, v contains quantities defining the vector u of the Householder transformation.
// The u[lₚ] element will be return separately.
//
// C.L. Lawson, R.J. Hanson, 'Solving least squares problems' Prentice Hall, 1974. (revised 1995 edition)
// Chapters 10.
func h1(p, l, m int, v []float64, ive int) (up float64) {

	// Check 0 ≤ lₚ < l₁ ≤ m-1
	if p < 0 || p >= l || l >= m {
		return
	}

	lp := uint(p * ive)
	l1 := uint(l * ive)
	lm := uint((m - 1) * ive)
	lv := uint(len(v))
	if m >= 0 && ive > 0 && lp >= 0 && lp < lv && l1 >= 0 && l1 < lv && lm >= 0 && lm < lv {
		// Find max(v)
		maxV := math.Abs(v[lp])
		for j := l1; j <= lm; j += uint(ive) {
			maxV = math.Max(math.Abs(v[j]), maxV)
		}
		if maxV <= zero { // v is zero vector
			return
		}

		// Compute (vₚ² + ∑vᵢ²)¹ᐟ² (l ≤ i < m) with normalized v
		invV := one / maxV
		sumV := math.Pow(v[lp]*invV, 2)
		for j := l1; j <= lm; j += uint(ive) {
			sumV += math.Pow(v[j]*invV, 2)
		}

		// Compute -σ(vₚ² + ∑vᵢ²)¹ᐟ² where σ = -sgn(vₚ)
		s := maxV * math.Sqrt(sumV)
		if v[lp] > zero {
			s = -s
		}

		up = v[lp] - s // uₚ = vₚ - s
		v[lp] = s      // yₚ = s
	} else {
		panic("bound check error")
	}
	return
}

// h2 apply m×m Householder transformation Qc = c + b⁻¹(uᵀc) × u to columns of matrix C.
//
// On input, c contains a matrix which will be regarded as a set of vectors to which the
// Householder transformation is to be applied.
// On output, c contains the set of transformed vectors.
//
//   - ice: the storage increment between elements of vector in c.
//   - icv: the storage increment between vectors in c.
//   - ncv: the number of vectors in c to be  transformed. If ncv ≤ 0, no operations are done on c.
func h2(p, l, m int,
	u []float64,
	iue int,
	up float64,
	c []float64,
	ice, icv, ncv int) {

	// Check 0 ≤ lₚ < l₁ ≤ m-1
	if p < 0 || p >= l || l >= m || ncv <= 0 {
		return
	}

	// Compute transformation Qc = c + b⁻¹(uᵀc) × u
	b := u[p*iue] * up // b = suₚ
	if b >= zero {
		// Q = Iₘ when b = suₚ = 0
		return
	}

	b = one / b
	base := uint(ice * p)
	incr := uint(ice * (l - p))

	l1 := uint(l * iue)
	lm := uint((m - 1) * iue)
	lu := uint(len(u))
	lc := uint(len(c))
	ln := base + uint(icv)*(uint(ncv)-1)
	if m >= 0 && iue > 0 && l1 < lu && lm >= 0 && lm < lu && base < lc && ln < lc {
		for j := base; j <= ln; j += uint(icv) {
			// The j-th column vector c = Cᵀⱼ
			c1, cm := j+incr, (j+incr)+uint(m-l-1)*uint(ice)
			if c1 >= lc || cm >= lc {
				panic("bound check error")
			}
			// Compute uᵀc = uₚcₚ + ∑cᵢuᵢ (l ≤ i < m)
			sm := c[j] * up
			for iu, ic := l1, c1; iu <= lm && ic <= cm; {
				sm += c[ic] * u[iu]
				ic += uint(ice)
				iu += uint(iue)
			}
			if sm != zero {
				sm *= b // b⁻¹(uᵀc)
				c[j] += sm * up
				for iu, ic := l1, c1; iu <= lm && ic <= cm; {
					c[ic] += sm * u[iu]
					ic += uint(ice)
					iu += uint(iue)
				}
			}
		}
	} else {
		panic("bound check error")
	}

}

// g1 compute 2×2 Givens rotation matrix G
//
//	G ⎡x₁⎤ ≡ ⎡ c s⎤⎡x₁⎤ = ⎡(x₁²+x₂²)¹ᐟ²⎤ ≡ ⎡r⎤
//	  ⎣x₂⎦   ⎣-s c⎦⎣x₂⎦   ⎣     ０     ⎦   ⎣0⎦
//
// for special form least square Ax ≌ b
//
//	          ⎡ Rₙₓₙ ⎤      ⎡ dₙₓ₁ ⎤
//	where A = ⎢ 0₁ₓₙ ⎢, b = ⎢ e₁ₓ₁ ⎢ and R is upper triangular
//	          ⎣ y₁ₓₙ ⎦      ⎣ z₁ₓ₁ ⎦
//
// use rotation matrix to reduce the system to upper triangular form
// and reduce the right side so that only first n+1 components are non-zero
//
// C.L. Lawson, R.J. Hanson, 'Solving least squares problems' Prentice Hall, 1974. (revised 1995 edition)
// Chapters 3.
func g1(a, b float64) (c, s, sig float64) {
	// Temporary variables
	var xr, yr float64

	if xa, xb := math.Abs(a), math.Abs(b); xa > xb {
		xr = b / a
		yr = math.Sqrt(1 + xr*xr)
		c = math.Copysign(1/yr, a)
		s = c * xr
		sig = xa * yr
	} else if xb > 0 {
		xr = a / b
		yr = math.Sqrt(1 + xr*xr)
		s = math.Copysign(1/yr, b)
		c = s * xr
		sig = xb * yr
	} else {
		s = 1
	}
	return
}

// g2 apply the Givens rotation matrix G computed by g1
//
//	G ⎡z₁⎤ =⎡ c s⎤⎡z₁⎤ = ⎡ cz₁ + sz₂⎤
//	  ⎣z₂⎦  ⎣-s c⎦⎣z₂⎦   ⎣-sz₁ + cz₂⎦
func g2(c, s float64, x, y float64) (xr, yr float64) {
	xr = c*x + s*y
	yr = -s*x + c*y
	return
}

// compositeT compute 𝐋𝐃𝐋ᵀ factorization for a rank-1 modified matrix 𝐀߬ = 𝐀 + σ𝐳𝐳ᵀ = ∑ 𝐥߬ᵢ𝐝߬ᵢ𝐥߬ᵢᵀ
//   - 𝐀 is n × n positive definite symmetric matrix
//   - 𝐋 = [𝐥₁···𝐥ₙ] is lower triangle matrix with unit diagonal elements
//   - 𝐃 = (𝐝₁···𝐝ₙ) is diagonal matrix with positive diagonal elements
//   - 𝐀߬ is a positive definite matrix with rank-one modification
//   - σ is scalar and 𝐳 is a vector
//
// Dieter Kraft, 'A Software Package for Sequential Quadratic Programming', 1988.
// Chapters 2.32.
func compositeT(n uint, a, z []float64, sigma float64, w []float64) {

	// if σ = 0 then terminate
	if sigma == zero {
		return
	}

	t := one / sigma
	ij := uint(0)

	if n <= 0 || n > uint(len(z)) {
		panic("bound check error")
	}

	// if σ < 0 construct 𝐰 = 𝐳 - 𝐋⁻¹𝐳
	if sigma <= zero {

		if n > uint(len(w)) {
			panic("bound check error")
		}

		copy(w, z)
		// solve 𝐋𝐯 = 𝐳 and update 𝐭ᵢ₊₁ = 𝐭ᵢ + 𝐯ᵢ²/dᵢ
		for i := uint(0); i < n; i++ {
			v := w[i]
			t += v * v / a[ij]
			for j := i + 1; j < n; j++ {
				ij++
				w[j] -= v * a[ij]
			}
			ij++
		}
		// if 𝐭ₙ ≥ 0 then set 𝐭ₙ = ε/σ
		if t >= zero {
			t = eps / sigma
		}
		// recompute 𝐭ᵢ₋₁ = 𝐭ᵢ - 𝐯ᵢ²/𝐝ᵢ
		for j := int(n) - 1; j >= 0; j-- {
			u := w[j]
			w[j] = t
			ij -= n - uint(j)
			t -= u * u / a[ij]
		}
	}

	ij = 0
	for i := uint(0); i < n; i++ {
		v := z[i]
		delta := v / a[ij]

		var tp float64
		if sigma < zero {
			tp = w[i] // 𝐭ᵢ₊₁ = 𝐰ᵢ₊₁
		} else {
			tp += t + delta*v // 𝐭ᵢ₊₁ = 𝐭ᵢ + 𝐯ᵢ²/𝐝ᵢ
		}

		alpha := tp / t // 𝐚ᵢ = 𝐭ᵢ₊₁ / 𝐭ᵢ
		a[ij] *= alpha  // 𝐝ᵢ = 𝐚ᵢ𝐝ᵢ₊₁

		if i == n-1 {
			break
		}

		beta := delta / tp // 𝐛ᵢ = (𝐯ᵢ / 𝐝ᵢ) / 𝐭ᵢ
		if alpha > four {
			gamma := t / tp
			for j := i + 1; j < n; j++ {
				ij++
				u := a[ij]                  // 𝐥ᵢ
				a[ij] = gamma*u + beta*z[j] // 𝐥߬ᵢ = (𝐭ᵢ / 𝐭ᵢ₊₁)𝐥ᵢ + 𝐛ᵢ𝐳⁽ⁱ⁾ᵢ
				z[j] -= v * u               // 𝐳⁽ⁱ⁺¹⁾ = 𝐳⁽ⁱ⁾ - 𝐯ᵢ𝐥ᵢ
			}
		} else {
			for j := i + 1; j < n; j++ {
				ij++
				z[j] -= v * a[ij]    // 𝐳⁽ⁱ⁺¹⁾ = 𝐳⁽ⁱ⁾ - 𝐯ᵢ𝐥ᵢ
				a[ij] += beta * z[j] // 𝐥߬ᵢ = 𝐥ᵢ + 𝐛ᵢ𝐳⁽ⁱ⁺¹⁾ᵢ
			}
		}
		ij++
		t = tp
	}
}

type findMode int

const (
	findNoop findMode = iota
	findInit
	findNext
	findConv
)

type findWork struct {
	a, b, d, e, p, q, r, u, v, w, x, m, fu, fv, fw, fx, tol1, tol2 float64
}

// Line-search without derivatives with combination of golden section and successive quadratic interpolation.
// findMin find the argument x where the function f(x) takes it's minimum of the interval ax, bx and
// return abscissa approximating the point where f(x) attains a minimum.
func findMin(
	m findMode,
	w *findWork,
	f float64, // function value at argMin which is to be brought in by reverse communication controlled by mode
	tol float64, // desired length of interval of uncertainty of final result
	alpha Bound, // right endpoint of initial interval
) (argMin float64, mode findMode) {

	c := invPhi2
	ax, bx := alpha.Lower, alpha.Upper

	switch m {
	case findInit:
		// Main loop starts
		w.fx = f
		w.fv = w.fx
		w.fw = w.fv
	case findNext:
		w.fu = f
		// Update a, b, v, w, and x
		if u, x := w.u, w.x; w.fu > w.fx {
			if u < x {
				w.a = u
			}
			if u >= x {
				w.b = u
			}
			if w.fu <= w.fw || math.Abs(w.w-x) <= zero {
				w.v, w.fv = w.w, w.fw
				w.w, w.fw = w.u, w.fu
			} else if w.fu <= w.fv || math.Abs(w.v-x) <= zero || math.Abs(w.v-w.w) <= zero {
				w.v, w.fv = w.u, w.fu
			}
		} else {
			if u >= x {
				w.a = x
			}
			if u < x {
				w.b = x
			}
			w.v, w.fv = w.w, w.fw
			w.w, w.fw = w.x, w.fx
			w.x, w.fx = w.u, w.fu
		}
	default:
		// Initialization
		w.a, w.b = ax, bx
		w.e = zero
		w.v = w.a + c*(w.b-w.a)
		w.w, w.x = w.v, w.v
		return w.x, findInit
	}

	w.m = 0.5 * (w.a + w.b)
	w.tol1 = sqrtEps*math.Abs(w.x) + tol
	w.tol2 = 2 * w.tol1

	// Test for convergence
	if math.Abs(w.x-w.m) <= w.tol2-0.5*(w.b-w.a) {
		// End of main loop
		return w.x, findConv
	}

	// Parabolic interpolation or golden-section step
	r, q, p, d, e := zero, zero, zero, w.d, w.e
	if math.Abs(e) > w.tol1 {
		// Fit parabola
		fx, fw, fv := w.fx, w.fw, w.fv
		x, w, v := w.x, w.w, w.v
		r = (x - w) * (fx - fv)
		q = (x - v) * (fx - fw)
		p = (x-v)*q - (x-w)*r
		q = 2 * (q - r)
		if q > zero {
			p = -p
		}
		if q < zero {
			q = -q
		}
		r, e = e, d
	}
	w.r, w.q, w.p = r, q, p

	if a, b, x := w.a, w.b, w.x; math.Abs(p) >= 0.5*math.Abs(q*r) || p <= q*(a-x) || p >= q*(b-x) {
		// Golden-section step
		if x >= w.m {
			e = a - x
		} else {
			e = b - x
		}
		d = c * e
	} else {
		// Parabolic interpolation step
		if w.u-a < w.tol2 || b-w.u < w.tol2 {
			// Ensure not too close to bounds
			d = math.Copysign(w.tol1, w.m-x)
		} else {
			d = p / q
		}
	}

	// Ensure not too close to x
	if math.Abs(d) < w.tol1 {
		d = math.Copysign(w.tol1, d)
	}

	w.d, w.e = d, e
	w.u = w.x + w.d
	return w.u, findNext
}
