package numdiff

import (
	"math"
	"reflect"
	"slices"
	"testing"
)

func objV2(x, y []float64) {
	y[0] = x[0] * math.Sin(x[1])
	y[1] = x[1] * math.Cos(x[0])
	y[2] = math.Pow(x[0], 3) * math.Pow(x[1], -0.5)
}

func jacV2(x []float64) []float64 {
	return []float64{
		math.Sin(x[1]), x[0] * math.Cos(x[1]),
		-x[1] * math.Sin(x[0]), math.Cos(x[0]),
		3 * math.Pow(x[0], 2) * math.Pow(x[1], -0.5), -0.5 * math.Pow(x[0], 3) * math.Pow(x[1], -1.5),
	}
}

func objZero(x, y []float64) {
	y[0] = x[0] * x[1]
	y[1] = math.Cos(x[0] * x[1])
}

func jacZero(x []float64) []float64 {
	return []float64{
		x[1], x[0],
		-x[1] * math.Sin(x[0]*x[1]), -x[0] * math.Sin(x[0]*x[1]),
	}
}

// Case Sources : https://github.com/scipy/scipy/blob/main/scipy/optimize/tests/test__numdiff.py (TestAdjustSchemeToBounds)
func TestAdjustToBnd(t *testing.T) {

	// test_no_bounds
	{
		x0 := slices.Repeat([]float64{0}, 3)
		h0 := slices.Repeat([]float64{0.01}, 3)
		dummy := make([]float64, 3)

		as := ApproxSpec{N: 3, M: 1}

		as.Method = Forward
		_ = as.Check(x0, dummy)
		copy(as.absStep, h0)
		as.adjustToBounds(x0, false)

		switch {
		case !relativeEqual(as.absStep, h0, 0):
			t.Fatal("unexpected adjust step")
		case len(as.oneSide) > 0:
			t.Fatal("unexpected side flag")
		}

		as.Method = Central
		_ = as.Check(x0, dummy)
		copy(as.absStep, h0)
		as.adjustToBounds(x0, false)

		switch {
		case !relativeEqual(as.absStep, h0, 0):
			t.Fatal("unexpected adjust step")
		case len(as.oneSide) != as.N || slices.Index(as.oneSide, true) != -1:
			t.Fatal("unexpected side flag")
		}
	}

	// test_with_bound
	{
		x0 := []float64{0, 0.85, -0.85}
		h0 := []float64{0.1, 0.1, -0.1}
		dummy := make([]float64, 3)

		as := ApproxSpec{N: 3, M: 1}
		as.Bounds = []Bound{{-1, 1}, {-1, 1}, {-1, 1}}

		as.Method = Forward
		_ = as.Check(x0, dummy)
		copy(as.absStep, h0)
		as.adjustToBounds(x0, true)

		switch {
		case !relativeEqual(as.absStep, h0, 0):
			t.Fatal("unexpected adjust step")
		case len(as.oneSide) > 0:
			t.Fatal("unexpected side flag")
		}

		as.Method = Central
		_ = as.Check(x0, dummy)
		copy(as.absStep, h0)
		as.adjustToBounds(x0, true)

		switch {
		case !relativeEqual(as.absStep, []float64{0.1, 0.1, 0.1}, 0):
			t.Fatal("unexpected adjust step")
		case len(as.oneSide) != as.N || slices.Index(as.oneSide, true) != -1:
			t.Fatal("unexpected side flag")
		}
	}

	// test_tight_bounds
	{
		x0 := []float64{0.0, 0.03}
		h0 := []float64{-0.1, -0.1}
		dummy := make([]float64, 2)

		as := ApproxSpec{N: 2, M: 1}
		as.Bounds = []Bound{{-0.03, 0.05}, {-0.03, 0.05}}

		as.Method = Forward
		_ = as.Check(x0, dummy)
		copy(as.absStep, h0)
		as.adjustToBounds(x0, true)

		switch {
		case !relativeEqual(as.absStep, []float64{0.05, -0.06}, 0):
			t.Fatal("unexpected adjust step")
		case len(as.oneSide) > 0:
			t.Fatal("unexpected side flag")
		}

		as.Method = Central
		_ = as.Check(x0, dummy)
		copy(as.absStep, h0)
		as.adjustToBounds(x0, true)

		switch {
		case !relativeEqual(as.absStep, []float64{0.03, -0.03}, 0):
			t.Fatal("unexpected adjust step")
		case !reflect.DeepEqual(as.oneSide, []bool{false, true}):
			t.Fatal("unexpected side flag")
		}
	}

}

// Case Sources : https://github.com/scipy/scipy/blob/main/scipy/optimize/tests/test__numdiff.py (test_absolute_step_sign)
func TestComputeAbsStp(t *testing.T) {

	x0 := []float64{1e-5, 0, 1, 1e5}
	dummy := make([]float64, 4)

	// auto select relative step
	for method, relStep := range map[Method]float64{
		Forward: sqrtEps,
		Central: cubeEps,
	} {

		expected := []float64{
			relStep,
			relStep * 1,
			relStep * 1,
			relStep * math.Abs(x0[3]),
		}

		as := ApproxSpec{N: 4, M: 1, Method: method}
		_ = as.Check(x0, dummy)

		as.absoluteStep(x0)
		if !relativeEqual(as.absStep, expected, 1e-12) {
			t.Fatal("unexpected abs step")
		}

		negX0 := make([]float64, len(x0))
		for i, v := range x0 {
			negX0[i] = -v
			expected[i] = math.Copysign(expected[i], -v)
		}

		as.absoluteStep(negX0)
		if !relativeEqual(as.absStep, expected, 1e-12) {
			t.Fatal("unexpected abs step")
		}
	}

	// user-specified relative step
	for _, relStep := range []float64{0.1, 1, 10, 100} {

		expected := []float64{
			relStep * x0[0],
			sqrtEps,
			relStep * x0[2],
			relStep * x0[3],
		}

		as := ApproxSpec{N: 4, M: 1, Method: Forward, RelStep: relStep}
		_ = as.Check(x0, dummy)

		as.absoluteStep(x0)
		if !relativeEqual(as.absStep, expected, 1e-12) {
			t.Fatal("unexpected abs step")
		}

		negX0 := make([]float64, len(x0))
		for i, v := range x0 {
			negX0[i] = -v
			expected[i] = math.Copysign(expected[i], -v)
		}

		as.absoluteStep(negX0)
		if !relativeEqual(as.absStep, expected, 1e-12) {
			t.Fatal("unexpected abs step")
		}
	}

}

// Case Sources : https://github.com/scipy/scipy/blob/main/scipy/optimize/tests/test__numdiff.py (test_absolute_step_sign)
func TestAbsStpSign(t *testing.T) {

	obj := func(x, y []float64) {
		y[0] = -math.Abs(x[0]+1) + math.Abs(x[1]+1)
	}

	x0 := []float64{-1, -1}
	grad := []float64{0, 0}

	as := ApproxSpec{N: 2, M: 1, Method: Forward, Object: obj, AbsStep: 1e-8}
	if err := as.Diff(x0, grad); err != nil {
		t.Fatal("abs sign failed", err)
	}
	if !relativeEqual(grad, []float64{-1.0, 1.0}, 1e-7) {
		t.Fatal("unexpected abs sign")
	}

	as = ApproxSpec{N: 2, M: 1, Method: Forward, Object: obj, AbsStep: -1e-8}
	if err := as.Diff(x0, grad); err != nil {
		t.Fatal("abs sign failed", err)
	}
	if !relativeEqual(grad, []float64{1.0, -1.0}, 1e-7) {
		t.Fatal("unexpected abs sign")
	}

	as = ApproxSpec{N: 2, M: 1, Method: Forward, Object: obj, AbsStep: 1e-8,
		Bounds: []Bound{{math.Inf(-1), -1}, {math.Inf(-1), -1}}}
	if err := as.Diff(x0, grad); err != nil {
		t.Fatal("abs sign failed", err)
	}
	if !relativeEqual(grad, []float64{1.0, -1.0}, 1e-7) {
		t.Fatal("unexpected abs sign")
	}

	as = ApproxSpec{N: 2, M: 1, Method: Forward, Object: obj, AbsStep: -1e-8,
		Bounds: []Bound{{-1, math.Inf(1)}, {-1, math.Inf(1)}}}
	if err := as.Diff(x0, grad); err != nil {
		t.Fatal("abs sign failed", err)
	}
	if !relativeEqual(grad, []float64{-1.0, 1.0}, 1e-7) {
		t.Fatal("unexpected abs sign")
	}
}

// Case Sources : https://github.com/scipy/scipy/blob/main/scipy/optimize/tests/test__numdiff.py
// (TestApproxDerivativesDense.test_scalar_scalar)
func TestScalar(t *testing.T) {

	x0 := []float64{1.0}
	obj := func(x, y []float64) {
		y[0] = math.Sinh(x[0])
	}

	jac1 := []float64{math.Cosh(x0[0])}
	jac2 := []float64{0}
	jac3 := []float64{0}

	as := ApproxSpec{N: 1, M: 1, Method: Forward, Object: obj}
	if err := as.Diff(x0, jac2); err != nil {
		t.Fatal("approx scalar failed", err)
	}
	as = ApproxSpec{N: 1, M: 1, Method: Central, Object: obj}
	if err := as.Diff(x0, jac3); err != nil {
		t.Fatal("approx scalar failed", err)
	}
	if !relativeEqual(jac2, jac1, 1e-6) {
		t.Fatal("unexpected approx scalar result")
	}
	if !relativeEqual(jac3, jac1, 1e-9) {
		t.Fatal("unexpected approx scalar result")
	}

	as = ApproxSpec{N: 1, M: 1, Method: Forward, Object: obj, AbsStep: 1.49e-8}
	if err := as.Diff(x0, jac2); err != nil {
		t.Fatal("approx scalar failed", err)
	}
	as = ApproxSpec{N: 1, M: 1, Method: Central, Object: obj, AbsStep: 1.49e-8}
	if err := as.Diff(x0, jac3); err != nil {
		t.Fatal("approx scalar failed", err)
	}
	if !relativeEqual(jac2, jac1, 1e-6) {
		t.Fatal("unexpected approx scalar result")
	}
	if !relativeEqual(jac3, jac1, 1e-6) {
		t.Fatal("unexpected approx scalar result")
	}

}

// Case Sources : https://github.com/scipy/scipy/blob/main/scipy/optimize/tests/test__numdiff.py
// (TestApproxDerivativesDense.test_scalar_vector)
func TestScalarVec(t *testing.T) {
	x0 := []float64{0.5}
	obj := func(x, y []float64) {
		y[0] = x[0] * x[0]
		y[1] = math.Tan(x[0])
		y[2] = math.Exp(x[0])
	}

	jac1 := []float64{
		2 * x0[0],
		1 / (math.Cos(x0[0]) * math.Cos(x0[0])),
		math.Exp(x0[0]),
	}

	jac2 := []float64{0, 0, 0}
	jac3 := []float64{0, 0, 0}

	as := ApproxSpec{N: 1, M: 3, Method: Forward, Object: obj}
	if err := as.Diff(x0, jac2); err != nil {
		t.Fatal("approx scalar failed", err)
	}
	as = ApproxSpec{N: 1, M: 3, Method: Central, Object: obj}
	if err := as.Diff(x0, jac3); err != nil {
		t.Fatal("approx scalar-vec failed", err)
	}
	if !relativeEqual(jac2, jac1, 1e-6) {
		t.Fatal("unexpected approx scalar-vec result")
	}
	if !relativeEqual(jac3, jac1, 1e-9) {
		t.Fatal("unexpected approx scalar-vec result")
	}

	as = ApproxSpec{N: 1, M: 3, Method: Forward, Object: obj, AbsStep: 1.49e-8}
	if err := as.Diff(x0, jac2); err != nil {
		t.Fatal("approx scalar-vec failed", err)
	}
	as = ApproxSpec{N: 1, M: 3, Method: Central, Object: obj, AbsStep: 1.49e-8}
	if err := as.Diff(x0, jac3); err != nil {
		t.Fatal("approx scalar-vec failed", err)
	}
	if !relativeEqual(jac2, jac1, 1e-6) {
		t.Fatal("unexpected approx scalar-vec result")
	}
	if !relativeEqual(jac3, jac1, 1e-9) {
		t.Fatal("unexpected approx scalar result")
	}

}

// Case Sources : https://github.com/scipy/scipy/blob/main/scipy/optimize/tests/test__numdiff.py
// (TestApproxDerivativesDense.test_vector_scalar)
func TestVecScalar(t *testing.T) {
	x0 := []float64{100.0, -0.5}
	obj := func(x, y []float64) {
		y[0] = math.Sin(x[0]*x[1]) * math.Log(x[0])
	}

	jac1 := []float64{
		x0[1]*math.Cos(x0[0]*x0[1])*math.Log(x0[0]) + math.Sin(x0[0]*x0[1])/x0[0],
		x0[0] * math.Cos(x0[0]*x0[1]) * math.Log(x0[0]),
	}

	jac2 := []float64{0, 0}
	jac3 := []float64{0, 0}

	as := ApproxSpec{N: 2, M: 1, Method: Forward, Object: obj}
	if err := as.Diff(x0, jac2); err != nil {
		t.Fatal("approx vec-scalar failed", err)
	}
	as = ApproxSpec{N: 2, M: 1, Method: Central, Object: obj}
	if err := as.Diff(x0, jac3); err != nil {
		t.Fatal("approx vec-scalar failed", err)
	}
	if !relativeEqual(jac2, jac1, 1e-6) {
		t.Fatal("unexpected approx vec-scalar result")
	}
	if !relativeEqual(jac3, jac1, 1e-7) {
		t.Fatal("unexpected approx vec-scalar result")
	}

	as = ApproxSpec{N: 2, M: 1, Method: Forward, Object: obj, AbsStep: 1.49e-8}
	if err := as.Diff(x0, jac2); err != nil {
		t.Fatal("approx vec-scalar failed", err)
	}
	as = ApproxSpec{N: 2, M: 1, Method: Central, Object: obj, AbsStep: 1.49e-8}
	if err := as.Diff(x0, jac3); err != nil {
		t.Fatal("approx vec-scalar failed", err)
	}
	if !relativeEqual(jac2, jac1, 1e-6) {
		t.Fatal("unexpected approx vec-scalar result")
	}
	if !relativeEqual(jac3, jac1, 1e-6) {
		t.Fatal("unexpected approx vec-scalar result")
	}

}

// Case Sources : https://github.com/scipy/scipy/blob/main/scipy/optimize/tests/test__numdiff.py
// (TestApproxDerivativesDense.test_vector_vector)
func TestVector(t *testing.T) {

	x0 := []float64{-100.0, 0.2}
	obj := objV2
	jac1 := jacV2(x0)
	jac2 := make([]float64, 6)
	jac3 := make([]float64, 6)

	as := ApproxSpec{N: 2, M: 3, Method: Forward, Object: obj}
	if err := as.Diff(x0, jac2); err != nil {
		t.Fatal("approx vector failed", err)
	}
	as = ApproxSpec{N: 2, M: 3, Method: Central, Object: obj}
	if err := as.Diff(x0, jac3); err != nil {
		t.Fatal("approx vector failed", err)
	}
	if !relativeEqual(jac1, jac2, 1e-5) {
		t.Fatal("unexpected approx vector result")
	}
	if !relativeEqual(jac1, jac3, 1e-6) {
		t.Fatal("unexpected approx vector result")
	}

	as = ApproxSpec{N: 2, M: 3, Method: Forward, Object: obj, RelStep: 1e-4}
	if err := as.Diff(x0, jac2); err != nil {
		t.Fatal("approx vector failed", err)
	}
	as = ApproxSpec{N: 2, M: 3, Method: Central, Object: obj, RelStep: 1e-4}
	if err := as.Diff(x0, jac3); err != nil {
		t.Fatal("approx vector failed", err)
	}
	if !relativeEqual(jac1, jac2, 1e-2) {
		t.Fatal("unexpected approx vector result")
	}
	if !relativeEqual(jac1, jac3, 1e-4) {
		t.Fatal("unexpected approx scalar-vec result")
	}

}

// Case Sources : https://github.com/scipy/scipy/blob/main/scipy/optimize/tests/test__numdiff.py
// (TestApproxDerivativesDense.test_with_bounds_2_point)
func TestBound2(t *testing.T) {

	bnd := []Bound{{-1, 1}, {-1, 1}}
	obj := objV2

	jac := make([]float64, 6)
	as := ApproxSpec{N: 2, M: 3, Object: obj, Bounds: bnd}
	if err := as.Diff([]float64{-2.0, 0.2}, jac); err == nil {
		t.Fatal("unexpected approx bound status")
	}

	x0 := []float64{-1.0, 1.0}
	jac0 := jacV2(x0)

	as = ApproxSpec{N: 2, M: 3, Method: Forward, Object: obj, Bounds: bnd}
	if err := as.Diff(x0, jac); err != nil {
		t.Fatal("approx bound failed", err)
	}

	if !relativeEqual(jac, jac0, 1e-6) {
		t.Fatal("unexpected approx vector result")
	}

}

// Case Sources : https://github.com/scipy/scipy/blob/main/scipy/optimize/tests/test__numdiff.py
// (TestApproxDerivativesDense.test_with_bounds_3_point)
func TestBound3(t *testing.T) {

	x0 := []float64{1.0, 2.0}
	obj := objV2

	jac0 := jacV2(x0)

	jac := make([]float64, 6)
	as := ApproxSpec{N: 2, M: 3, Method: Central, Object: obj}
	if err := as.Diff(x0, jac); err != nil {
		t.Fatal("approx bound failed", err)
	}
	if !relativeEqual(jac, jac0, 1e-9) {
		t.Fatal("unexpected approx bound result")
	}

	as = ApproxSpec{N: 2, M: 3, Method: Central, Object: obj,
		Bounds: []Bound{{1, math.Inf(1)}, {1, math.Inf(1)}}}
	if err := as.Diff(x0, jac); err != nil {
		t.Fatal("approx bound failed", err)
	}
	if !relativeEqual(jac, jac0, 1e-9) {
		t.Fatal("unexpected approx bound result")
	}

	as = ApproxSpec{N: 2, M: 3, Method: Central, Object: obj,
		Bounds: []Bound{{math.Inf(-1), 2}, {math.Inf(-1), 2}}}
	if err := as.Diff(x0, jac); err != nil {
		t.Fatal("approx bound failed", err)
	}
	if !relativeEqual(jac, jac0, 1e-9) {
		t.Fatal("unexpected approx bound result")
	}

	as = ApproxSpec{N: 2, M: 3, Method: Central, Object: obj,
		Bounds: []Bound{{1, 2}, {1, 2}}}
	if err := as.Diff(x0, jac); err != nil {
		t.Fatal("approx bound failed", err)
	}
	if !relativeEqual(jac, jac0, 1e-9) {
		t.Fatal("unexpected approx bound result")
	}

}

// Case Sources : https://github.com/scipy/scipy/blob/main/scipy/optimize/tests/test__numdiff.py
// (TestApproxDerivativesDense.test_tight_bounds)
func TestTightBnd(t *testing.T) {

	x0 := []float64{10.0, 10.0}
	bnd := []Bound{{x0[0] - 3e-9, x0[0] + 3e-9}, {x0[1] - 3e-9, x0[1] + 3e-9}}

	obj := func(x, y []float64) {
		y[0] = x[0] * math.Sin(x[1])
		y[1] = x[1] * math.Cos(x[0])
		y[2] = math.Pow(x[0], 3) * math.Pow(x[1], -0.5)
	}

	jac0 := []float64{
		math.Sin(x0[1]), x0[0] * math.Cos(x0[1]),
		-x0[1] * math.Sin(x0[0]), math.Cos(x0[0]),
		3 * math.Pow(x0[0], 2) * math.Pow(x0[1], -0.5), -0.5 * math.Pow(x0[0], 3) * math.Pow(x0[1], -1.5),
	}

	jac := make([]float64, 6)
	as := ApproxSpec{N: 2, M: 3, Method: Forward, Object: obj, Bounds: bnd}
	if err := as.Diff(x0, jac); err != nil {
		t.Fatal("approx tight-bound failed", err)
	}
	if !relativeEqual(jac, jac0, 1e-6) {
		t.Fatal("unexpected approx tight-bound result")
	}

	as = ApproxSpec{N: 2, M: 3, Method: Forward, Object: obj, Bounds: bnd, RelStep: 1e-6}
	if err := as.Diff(x0, jac); err != nil {
		t.Fatal("approx tight-bound failed", err)
	}
	if !relativeEqual(jac, jac0, 1e-6) {
		t.Fatal("unexpected approx tight-bound result")
	}

	as = ApproxSpec{N: 2, M: 3, Method: Central, Object: obj, Bounds: bnd}
	if err := as.Diff(x0, jac); err != nil {
		t.Fatal("approx tight-bound failed", err)
	}
	if !relativeEqual(jac, jac0, 1e-6) {
		t.Fatal("unexpected approx tight-bound result")
	}

	as = ApproxSpec{N: 2, M: 3, Method: Central, Object: obj, Bounds: bnd, RelStep: 1e-6}
	if err := as.Diff(x0, jac); err != nil {
		t.Fatal("approx tight-bound failed", err)
	}
	if !relativeEqual(jac, jac0, 1e-6) {
		t.Fatal("unexpected approx tight-bound result")
	}

}

// Case Sources : https://github.com/scipy/scipy/blob/main/scipy/optimize/tests/test__numdiff.py
// (TestApproxDerivativesDense.test_bound_switches)
func TestSwitchBnd(t *testing.T) {

	bnd := []Bound{{-1e-8, 1e-8}}

	obj := func(x, y []float64) {
		if math.Abs(x[0]) <= 1e-8 {
			y[0] = x[0]
		} else {
			y[0] = math.NaN()
		}
	}
	jac := func(x []float64) []float64 {
		if math.Abs(x[0]) <= 1e-8 {
			return []float64{1}
		} else {
			return []float64{math.NaN()}
		}
	}

	x0 := []float64{0}
	jac1 := jac(x0)
	jac2 := []float64{0}
	jac3 := []float64{0}

	as := ApproxSpec{N: 1, M: 1, Method: Forward, Object: obj, Bounds: bnd}
	if err := as.Diff(x0, jac2); err != nil {
		t.Fatal("approx switch-bound failed", err)
	}
	as = ApproxSpec{N: 1, M: 1, Method: Central, Object: obj, Bounds: bnd}
	if err := as.Diff(x0, jac3); err != nil {
		t.Fatal("approx switch-bound failed", err)
	}
	if !relativeEqual(jac2, jac1, 1e-6) {
		t.Fatal("unexpected approx switch-bound result")
	}
	if !relativeEqual(jac3, jac1, 1e-9) {
		t.Fatal("unexpected approx switch-bound result")
	}

	x0 = []float64{1e-8}
	jac1 = jac(x0)

	as = ApproxSpec{N: 1, M: 1, Method: Forward, Object: obj, Bounds: bnd, RelStep: 1e-6}
	if err := as.Diff(x0, jac2); err != nil {
		t.Fatal("approx switch-bound failed", err)
	}
	as = ApproxSpec{N: 1, M: 1, Method: Central, Object: obj, Bounds: bnd, RelStep: 1e-6}
	if err := as.Diff(x0, jac3); err != nil {
		t.Fatal("approx switch-bound failed", err)
	}
	if !relativeEqual(jac2, jac1, 1e-6) {
		t.Fatal("unexpected approx switch-bound result")
	}
	if !relativeEqual(jac3, jac1, 1e-9) {
		t.Fatal("unexpected approx switch-bound result")
	}

}

// Case Sources : https://github.com/scipy/scipy/blob/main/scipy/optimize/tests/test__numdiff.py
// (TestApproxDerivativesDense.test_check_derivative)
func TestAccuracy(t *testing.T) {

	checkDerivative := func(
		n, m int, x0 []float64,
		fun func(x, y []float64),
		jac func(x []float64) []float64) float64 {

		jacTest := jac(x0)
		jacDiff := make([]float64, n*m)

		approx := ApproxSpec{N: n, M: m, Method: Central, Object: fun}
		if err := approx.Diff(x0, jacDiff); err != nil {
			panic(err)
		}

		maxErr := 0.0
		for i := 0; i < n*m; i++ {
			absErr := math.Abs(jacTest[i] - jacDiff[i])
			absErr /= math.Max(1, math.Abs(jacDiff[i]))
			if absErr > maxErr {
				maxErr = absErr
			}
		}
		return maxErr
	}

	x0 := []float64{-10.0, 10}
	acc := checkDerivative(2, 3, x0, objV2, jacV2)
	if acc > 1e-9 {
		t.Fatal("approx accuracy not enough")
	}

	x0 = []float64{0, 0}
	acc = checkDerivative(2, 2, x0, objZero, jacZero)
	if acc > 0 {
		t.Fatal("approx accuracy not enough")
	}

}

func relativeEqual[T float64 | []float64](a, b T, tol float64) bool {
	equalWithinRel := func(a, b float64) bool {
		if a == b {
			return true
		}
		delta := math.Abs(a - b)
		return delta/math.Max(math.Abs(a), math.Abs(b)) <= tol
	}
	switch reflect.TypeFor[T]().Kind() {
	case reflect.Float64:
		return equalWithinRel(any(a).(float64), any(b).(float64))
	case reflect.Slice:
		a, b := any(a).([]float64), any(b).([]float64)
		if len(a) != len(b) {
			return false
		}
		for i, a := range a {
			if !equalWithinRel(a, b[i]) {
				return false
			}
		}
		return true
	default:
		panic("unknown type")
	}
}
