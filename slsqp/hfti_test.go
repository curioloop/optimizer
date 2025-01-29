// Copyright ©2025 curioloop. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slsqp

import (
	"testing"
)

// Origin: https://www.netlib.org/lawson-hanson/all (PROG2)
// Reference: https://people.math.sc.edu/Burkardt/f_src/lawson/lawson.html
func TestHFTI(t *testing.T) {

	const (
		mda = 8
		mdb = 8
		nb  = 1
	)

	a := make([]float64, mda*mda)
	b := make([]float64, mdb*nb)
	g := make([]float64, mda)
	h := make([]float64, mda)
	ip := make([]int, mda)
	srsmsq := make([]float64, nb)

	t.Log("Demonstrate the algorithms HFTI and COV.")

	var gen randGen

	for _, anoise := range []float64{zero, 0.0001} {

		gen.next(-one)

		var anorm, tau float64
		if anoise == zero {
			tau = 0.5
		} else {
			anorm = 500.0
			tau = anorm * anoise * 10.0
		}

		t.Logf("  Use a relative noise level of %.6f\n", anoise)
		t.Logf("  The matrix norm is approximately %.6f\n", anorm)
		t.Logf("  The absolute pseudorank tolerance is %.6f\n", tau)

		for mn1 := 1; mn1 <= 6; mn1 += 5 {
			for m := mn1; m <= mn1+2; m++ {
				for n := mn1; n <= mn1+2; n++ {

					t.Logf("\n  M = %d\n  N = %d\n", m, n)

					for i := 0; i < m; i++ {
						for j := 0; j < n; j++ {
							a[i+mda*j] = gen.next(anoise)
						}
					}

					for i := 0; i < m; i++ {
						b[i] = gen.next(anoise)
					}

					krank := HFTI(a, mda, m, n, b, mdb, nb, tau, srsmsq, h, g, ip)

					t.Logf("\n  Pseudorank = %d\n", krank)
					t.Log("  Estimated parameters X = A**(+)*B from HFTI:")
					for i := 0; i < n; i++ {
						t.Logf("%.6f ", b[i])
					}
					t.Logf("\n  Residual norm = %.6f\n", srsmsq[0])

					if n <= krank {
						// Covariance matrix computation
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

						for ii := 1; ii <= n; ii++ {
							i := n - ii
							if ip[i] != i {
								k := ip[i]

								temp := a[i+mda*i]
								a[i+mda*i] = a[k+mda*k]
								a[k+mda*k] = temp

								// Swap rows and columns
								for l := 0; l < i; l++ {
									temp = a[l+mda*i]
									a[l+mda*i] = a[l+mda*k]
									a[l+mda*k] = temp
								}
								for l := i + 1; l < k; l++ {
									temp = a[i+mda*l]
									a[i+mda*l] = a[l+mda*k]
									a[l+mda*k] = temp
								}
								for l := k + 1; l < n; l++ {
									temp = a[i+mda*l]
									a[i+mda*l] = a[k+mda*l]
									a[k+mda*l] = temp
								}
							}
						}

						t.Log("Unscaled covariance matrix of estimated parameters computed by COV:")
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
}
