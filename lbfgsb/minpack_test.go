// Copyright ©2025 curioloop. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lbfgsb

import (
	"math"
	"math/rand/v2"
	"testing"
)

type searchIter struct {
	Phi, Der   func(float64) float64
	Phi0, Der0 float64
	ctx        SearchCtx
	tol        SearchTol
}

func (d *searchIter) call(alpha1 float64, maxIter int) (stp, phi1, phi0 float64, task SearchTask) {

	phi0 = d.Phi0
	phi1 = phi0

	der0 := d.Der0
	der1 := der0

	stp = alpha1
	task = SearchStart
	for maxIter > 0 {
		stp, task = ScalarSearch(phi1, der1, stp, task, &d.tol, &d.ctx)
		if math.IsInf(stp, 0) {
			task = SearchWarn
			stp = math.NaN()
			break
		}
		if task == SearchFG {
			alpha1 = stp
			phi1 = d.Phi(stp)
			der1 = d.Der(stp)
		} else {
			break
		}
		maxIter--
	}
	if maxIter == 0 {
		// maxIter reached, the line search did not converge
		panic("STP NOT CONVERGE")
	}
	if task&(SearchError|SearchWarn) > 0 {
		stp = math.NaN()
	}
	return
}

func scalarSearchWolfe1(phi, der func(float64) float64, oldPhi0 float64) (stp, phi1, phi0 float64) {

	phi0, der0 := phi(0), der(0)

	alpha1 := 1.0
	if !math.IsNaN(oldPhi0) && der0 != 0 {
		alpha1 = math.Min(1, 1.01*2*(phi0-oldPhi0)/der0)
		if alpha1 < 0 {
			alpha1 = 1
		}
	}

	c1 := 1e-4
	c2 := 0.9
	if !(0 < c1 && c1 < c2 && c2 < 1) {
		panic("'c1' and 'c2' do not satisfy '0 < c1 < c2 < 1'.")
	}

	search := searchIter{
		Phi:  phi,
		Der:  der,
		Phi0: phi0,
		Der0: der0,
		tol: SearchTol{
			Alpha: c1,
			Beta:  c2,
			Eps:   1e-14,
			Lower: 1e-8,
			Upper: 50,
		},
	}

	stp, phi1, phi0, _ = search.call(alpha1, 100)
	return
}

func wolfeConditionHold(s float64, phi, der func(float64) float64) bool {
	phi0 := phi(0)
	der0 := der(0)

	phi1 := phi(s)
	der1 := der(s)

	c1 := 1e-4
	c2 := 0.9

	if phi1 > phi0+c1*s*der0 {
		return false
	}
	if math.Abs(der1) > math.Abs(c2*der0) {
		return false
	}
	return true
}

func TestScalarSearch(t *testing.T) {

	FGs := [][2]func(float64) float64{
		{
			func(s float64) float64 { return -s - math.Pow(s, 3) + math.Pow(s, 4) },
			func(s float64) float64 { return -1 - 3*math.Pow(s, 2) + 4*math.Pow(s, 3) },
		},
		{
			func(s float64) float64 { return math.Exp(-4*s) + math.Pow(s, 2) },
			func(s float64) float64 { return -4*math.Exp(-4*s) + 2*s },
		},
		{
			func(s float64) float64 { return -math.Sin(10 * s) },
			func(s float64) float64 { return -10 * math.Cos(10*s) },
		},
	}

	for _, fg := range FGs {
		phi, der := fg[0], fg[1]
		for i := 0; i < 3; i++ {
			oldPhi0 := rand.Float64()
			s, phi1, phi0 := scalarSearchWolfe1(phi, der, oldPhi0)

			pass := ulpDiff(phi0, phi(0)) < 50 &&
				ulpDiff(phi1, phi(s)) < 50 &&
				wolfeConditionHold(s, phi, der)

			if !pass {
				t.Fatal("scalar search failed")
			}

		}
	}
}

func ulpDiff(a, b float64) int64 {
	if a == b {
		return 0
	}
	if math.IsNaN(a) || math.IsNaN(b) {
		return math.MaxInt64
	}
	if math.IsInf(a, 0) || math.IsInf(b, 0) {
		if a == b {
			return 0
		}
		return math.MaxInt64
	}
	aInt := math.Float64bits(a)
	bInt := math.Float64bits(b)
	if aInt>>63 != bInt>>63 {
		return math.MaxInt64
	}
	diff := int64(aInt) - int64(bInt)
	if diff < 0 {
		return -diff
	}
	return diff
}
