// Copyright ©2025 curioloop. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slsqp

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

// Origin: https://www.netlib.org/lawson-hanson/all (PROG1)
// Reference: https://people.math.sc.edu/Burkardt/f_src/lawson/lawson.html
func TestH12(t *testing.T) {

	const mda = 8
	var gen randGen

	a := make([]float64, mda*mda)
	b := make([]float64, mda)
	h := make([]float64, mda)

	t.Log("  HFT factors a least squares problem;")
	t.Log("  HS1 solves a factored least squares problem;")
	t.Log("  COV computes the associated covariance matrix.")
	t.Log("  No checking will be made for rank deficiency in this test.")

	for _, anoise := range []float64{zero, 0.0001} {

		gen.next(-one)

		if anoise == zero {
			t.Log("\n  No noise used in matrix generation.")
		} else {
			t.Logf("\n  Matrix generation noise level = %.6f\n", anoise)
		}

		for mn1 := 1; mn1 <= 6; mn1 += 5 {
			mn2 := mn1 + 2

			for m := mn1; m <= mn2; m++ {
				for n := mn1; n <= m; n++ {
					t.Logf("\n\n   M   N\n%4d%4d\n", m, n)

					for i := 0; i < m; i++ {
						for j := 0; j < n; j++ {
							a[i+mda*j] = gen.next(anoise)
						}
					}

					for i := 0; i < m; i++ {
						b[i] = gen.next(anoise)
					}

					//fmt.Println(formatMat(m, n, a, mda))
					//fmt.Println(formatMat(1, n, b, 1))

					if m < n {
						continue
					}

					// Apply HFT algorithm
					for j := 0; j < n; j++ {
						k := min(j+1, n-1)
						h[j] = h1(j, j+1, m, a[j*mda:], 1)
						h2(j, j+1, m, a[j*mda:], 1, h[j], a[k*mda:], 1, mda, n-1-j)
					}

					// Apply HS1 algorithm
					for j := 0; j < n; j++ {
						h2(j, j+1, m, a[j*mda:], 1, h[j], b[:m], 1, 1, 1)
					}

					// Solve triangular system
					for k := 0; k < n; k++ {
						i := n - k - 1
						sm := zero
						if l := n - (i + 1); l > 0 {
							sm = ddot(n-(i+1), a[i+8*(i+1):], 8, b[i+1:], 1)
						}
						if a[i+mda*i] == zero {
							t.Log("\n  Terminating this case.")
							t.Log("  A divisor is exactly zero.")
							continue
						}
						b[i] = (b[i] - sm) / a[i+mda*i]
					}

					// Compute residual norm
					srsmsq := zero
					for j := n; j < m; j++ {
						srsmsq += b[j] * b[j]
					}
					srsmsq = math.Sqrt(srsmsq)

					// Begin COV algorithm
					for j := 0; j < n; j++ {
						a[j+mda*j] = one / a[j+mda*j]
					}

					for i := 0; i < n-1; i++ {
						for j := i + 1; j < n; j++ {
							sm := zero
							for l := i; l < j; l++ {
								sm += a[i+mda*l] * a[l+mda*j]
							}
							a[i+mda*j] = -sm * a[j+mda*j]
						}
					}

					for i := 0; i < n; i++ {
						for j := i; j < n; j++ {
							sm := zero
							for l := j; l < n; l++ {
								sm += a[i+mda*l] * a[j+mda*l]
							}
							a[i+mda*j] = sm
						}
					}

					// Print results
					t.Log("  Estimated parameters, x = a**(+)*b, computed by 'hft, hs1'")
					for i := 0; i < n; i++ {
						t.Logf("%6d %.7e\n", i+1, b[i])
					}
					t.Logf("  Residual length = %.4e\n", srsmsq)
					t.Log("  Covariance matrix (unscaled) of estimated parameters:")
					for i := 0; i < n; i++ {
						for j := i; j < n; j++ {
							t.Logf("%3d%3d %.7e\n", i+1, j+1, a[i+mda*j])
						}
					}
				}
			}
		}
	}
}

// generate a random value with noise added.
type randGen struct {
	i, j int
	aj   float64
}

// generate next random value with noise added.
// anoise determines the level of "noise" to be added to the data.
func (g *randGen) next(anoise float64) float64 {

	const (
		mi = 891
		mj = 457
	)

	if anoise < zero {
		g.i = 5
		g.j = 7
		g.aj = zero
		return zero
	}

	// The sequence of values of J is bounded between 1 and 996.
	// If initial j = 1,2,3,4,5,6,7,8, or 9, the period is 332.
	if anoise > zero {
		g.j = g.j * mj
		g.j = g.j - 997*(g.j/997)
		g.aj = float64(g.j - 498)
	}

	// The sequence of values of I is bounded between 1 and 999.
	// If initial i = 1,2,3,6,7, or 9, the period will be 50.
	// If initial i = 4 or 8, the period will be 25.
	// If initial i = 5, the period will be 10.
	g.i = g.i * mi
	g.i = g.i - 1000*(g.i/1000)
	return float64(g.i-500) + g.aj*anoise
}

func formatMat(rows, cols int, data []float64, stride int) string {
	var sb strings.Builder
	for i := 0; i < rows; i++ {
		if i == 0 {
			sb.WriteString("⎡")
		} else if i == rows-1 {
			sb.WriteString("⎣")
		} else {
			sb.WriteString("⎢")
		}
		for j := 0; j < cols; j++ {
			sb.WriteString(fmt.Sprintf(" %g", data[i+stride*j]))
			if j < cols-1 {
				sb.WriteString(" ")
			}
		}
		if i == 0 {
			sb.WriteString(" ⎤\n")
		} else if i == rows-1 {
			sb.WriteString(" ⎦\n")
		} else {
			sb.WriteString(" ⎥\n")
		}
	}
	return sb.String()
}
