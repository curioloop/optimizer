package numdiff

import (
	"errors"
	"math"
)

var sqrtEps = math.Sqrt(math.Nextafter(1, 2) - 1)
var cubeEps = math.Pow(math.Nextafter(1, 2)-1, float64(1)/3)

type Method int

const (
	// Forward use the first order accuracy forward difference.
	Forward Method = iota
	// Central use central difference in interior points and the second order accuracy
	// forward or backward difference near the boundary.
	Central
)

type Bound [2]float64

// ApproxSpec represents a numerical differentiation algorithms to estimate the derivative of a mathematical function.
//
// # Reference:
//
//   - https://en.wikipedia.org/wiki/Finite_difference
//   - https://github.com/scipy/scipy/blob/main/scipy/optimize/_numdiff.py
//
// # License
//
//   - https://github.com/scipy/scipy/blob/main/LICENSE.txt
type ApproxSpec struct {
	N, M int
	// Function of which to estimate the derivatives.
	// The argument x passed to this function is an n-vector.
	// The result is store in an m-vector y.
	Object func(x, y []float64)
	// Finite difference method to use.
	Method Method
	// Lower and upper bounds on independent variables.
	// Use it to limit the range of function evaluation.
	Bounds []Bound
	// Relative step size used to compute absolute step size.
	// The default absolute step size is computed as h = RelStep * sign(x0) * max(1, abs(x0)) with RelStep being selected automatically.
	// Otherwise, absolute step size is computed as h = RelStep * sign(x0) * abs(x0) when RelStep is provided.
	RelStep float64
	// Absolute step size to use, possibly adjusted to fit into the bounds.
	// The RelStep is used when AbsStep is not provide.
	// For Central method the sign of AbsStep is ignored.
	AbsStep float64
	// Don't check if x0 is out of bounds.
	NotChkBnd bool
	// Whether transpose the Jacobian matrix.
	TransJac bool
	approxCtx
}

type approxCtx struct {
	f0, fx  []float64
	absStep []float64
	oneSide []bool
}

// Check the parameters and initialize approxCtx.
func (as *ApproxSpec) Check(x0, diff []float64) (err error) {

	switch {
	case as.N <= 0 || as.M <= 0:
		err = errors.New("negative dimensions")
	case as.Method != Forward && as.Method != Central:
		err = errors.New("unknown method")
	case as.Object == nil:
		err = errors.New("object function is required")
	case as.N != len(x0):
		return errors.New("invalid x0 dimensions")
	case as.N*as.M != len(diff):
		return errors.New("invalid diff dimensions")
	}

	if as.Bounds != nil {
		if len(as.Bounds) != len(x0) {
			err = errors.New("invalid bound dimension")
		} else {
			for i, bound := range as.Bounds {
				if math.IsNaN(bound[0]) {
					bound[0] = math.Inf(-1)
				}
				if math.IsNaN(bound[1]) {
					bound[1] = math.Inf(1)
				}
				if bound[0] > bound[1] {
					err = errors.New("invalid bound range")
					break
				}
				if !as.NotChkBnd && (x0[i] < bound[0] || x0[i] > bound[1]) {
					err = errors.New("x0 violates bound constraints")
					break
				}
			}
		}
	}

	if len(as.fx) != as.M*(int(as.Method)+1) {
		as.f0 = make([]float64, as.M)
		as.fx = make([]float64, as.M*(int(as.Method)+1))
	}
	if len(as.absStep) != as.N {
		as.absStep = make([]float64, as.N)
	}
	if len(as.oneSide) != as.N*int(as.Method) {
		as.oneSide = make([]bool, as.N*int(as.Method))
	}
	return
}

// Diff calculate approximation of derivatives by finite differences.
func (as *ApproxSpec) Diff(x0, diff []float64) error {

	if err := as.Check(x0, diff); err != nil {
		return err
	}

	bnd := false
	for _, bound := range as.Bounds {
		l, u := bound[0], bound[1]
		if bnd = !(math.IsInf(l, 0) && math.IsInf(u, 0)); bnd {
			break
		}
	}

	as.absoluteStep(x0)
	as.adjustToBounds(x0, bnd)

	if as.Method == Central {
		as.approxCentral(x0, diff)
	} else {
		as.approxForward(x0, diff)
	}

	return nil
}

func (as *ApproxSpec) adjustToBounds(x0 []float64, bnd bool) {
	h, o := as.absStep, as.oneSide
	if as.Method == Central {
		for i, v := range h {
			h[i] = math.Abs(v)
		}
		for i := range o {
			o[i] = false
		}
	}

	if !bnd {
		return
	}

	b := as.Bounds
	if len(x0) != len(b) || len(x0) != len(h) {
		panic("bound check error")
	}

	if as.Method == Forward {
		for i, x0 := range x0 {
			lb, ub := b[i][0], b[i][1]
			ld, ud := x0-lb, ub-x0
			h0 := h[i]
			x := x0 + h0
			violated := x < lb || x > ub
			fitting := math.Abs(h[i]) < math.Max(ld, ud)
			if violated && fitting {
				h[i] = -h0
			} else if !fitting {
				if ud >= ld {
					h[i] = ud
				} else if ud < ld {
					h[i] = -ld
				}
			}
		}
	} else {
		if len(x0) != len(o) {
			panic("bound check error")
		}
		for i, x0 := range x0 {
			lb, ub := b[i][0], b[i][1]
			ld, ud := x0-lb, ub-x0
			central := ld >= h[i] && ud >= h[i]
			if !central {
				if ud >= ld {
					h[i] = math.Min(h[i], 0.5*ud)
					o[i] = true
				} else if ud < ld {
					h[i] = -math.Min(h[i], 0.5*ld)
					o[i] = true
				}
			}
			minDist := math.Min(ud, ld)
			adjCent := !central && math.Abs(h[i]) <= minDist
			if adjCent {
				h[i] = minDist
				o[i] = false
			}
		}
	}

}

func (as *ApproxSpec) absoluteStep(x0 []float64) {
	h := as.absStep
	if len(h) != len(x0) {
		panic("bound check error")
	}

	var eps float64
	switch as.Method {
	case Forward:
		eps = sqrtEps
	case Central:
		eps = cubeEps
	default:
		panic("unknown method")
	}

	abs := as.AbsStep
	rel := as.RelStep
	if abs == 0 && rel == 0 {
		for i, v := range x0 {
			h[i] = math.Copysign(eps, v) * math.Max(1.0, math.Abs(v))
		}
	} else {
		for i, v := range x0 {
			s := abs
			if s == 0 {
				s = math.Copysign(rel, v) * math.Abs(v)
			}
			d := (v + s) - v
			if d == 0 {
				s = math.Copysign(eps, v) * math.Max(1.0, math.Abs(v))
			}
			h[i] = s
		}
	}
}

func (as *ApproxSpec) approxForward(x0, df []float64) {

	f0, fx, h, n := as.f0, as.fx, as.absStep, as.N
	if len(h) != len(x0) || len(f0) != len(fx) {
		panic("bound check error")
	}

	fun := as.Object
	fun(x0, as.f0)
	for i, s := range h {
		t := x0[i]
		x0[i] = t + s
		fun(x0, fx)
		d := 1.0 / s
		for j := range f0 {
			df[i+j*n] = (fx[j] - f0[j]) * d
		}
		x0[i] = t
	}
}

func (as *ApproxSpec) approxCentral(x0, df []float64) {

	f0, h, o, n, m := as.f0, as.absStep, as.oneSide, as.N, as.M
	f1, f2 := as.fx[:m], as.fx[m:]
	if len(h) != len(x0) || len(h) != len(o) || len(f0) != len(f1) || len(f0) != len(f2) {
		panic("bound check error")
	}

	fun := as.Object
	fun(x0, as.f0)
	for i, s := range h {
		x := x0[i]
		d := 1.0 / (2 * s)
		if o[i] {
			x0[i] = x + s
			fun(x0, f1)
			x0[i] = x + 2*s
			fun(x0, f2)
			if !as.TransJac {
				for j := range f0 {
					df[i+j*n] = (4*f1[j] - 3*f0[j] - f2[j]) * d
				}
			} else {
				t := df[i*m : (i+1)*m]
				for j := range f0 {
					t[j] = (4*f1[j] - 3*f0[j] - f2[j]) * d
				}
			}
		} else {
			x0[i] = x - s
			fun(x0, f1)
			x0[i] = x + s
			fun(x0, f2)
			if !as.TransJac {
				for j := range f0 {
					df[i+j*n] = (f2[j] - f1[j]) * d
				}
			} else {
				t := df[i*m : (i+1)*m]
				for j := range f0 {
					t[j] = (f2[j] - f1[j]) * d
				}
			}
		}
		x0[i] = x
	}
}
