# Constrained Optimization in Pure Go

A pure Go implementation of constrained optimization algorithms **L-BFGS-B** and **SLSQP**. 

- **Pure Go**: No dependency on external libraries or C/Fortran code.
- **No External Dependencies**: Completely self-contained with zero third-party dependencies.
- **Detailed Comments**: Making it a great resource for learning the underlying optimization principles.

## L-BFGS-B

The L-BFGS-B algorithm is a quasi-Newton optimization algorithm that solves bound-constrained optimization problems. 
It is particularly useful in unconstrained and bounded optimization over large parameter spaces.

This Go implementation follows the structure of the Fortran code from  [L-BFGS-B](https://users.iems.northwestern.edu/~nocedal/lbfgs.html) (version 3.0).
The license can be found in the distribution file.

```go
package main

import (
	"fmt"
	"github.com/curioloop/optimizer/lbfgsb"
)


func main() {

	const (
		n = 2 // Define dimension of problem as n.
		m = 1 // Define number of L-BFGS corrections as m.
	)

	// Define the objective function and its gradient.
	// f(x) = x[0]^2 + x[1]^2,
	// g(x) = [2*x[0], 2*x[1]]
	eval := func(x []float64, g []float64) (f float64) {
		g[0], g[1] = 2*x[0], 2*x[1]  // gradient of objective function
		return x[0]*x[0] + x[1]*x[1] // calculate objective function
	}

	// Define termination conditions for the optimizer.
	stop := lbfgsb.Termination{
		MaxIterations:     5,
		MaxComputations:   5,
		MaxEvaluations:    5,
		EpsAccuracyFactor: 1e7,
		ProjGradTolerance: 1e-5,
	}

	// Define the bounds for the variables in the optimization problem.
	// Each element defines a lower and upper bound for a variable.
	bounds := []lbfgsb.Bound{
		{Lower: -5, Upper: 5},
		{Lower: -5, Upper: 5},
	}

	// Define a logger to capture and report optimization details.
	// In this case, we set the log level to LogLast to capture the last log entry.
	logger := &lbfgsb.Logger{
		Level: lbfgsb.LogLast,
	}

	// Initial guess for the optimization variables.
	// Here, the initial values for x[0] and x[1] are both set to 1.0.
	x0 := []float64{1.0, 1.0}

	// Initialize the optimizer.
	problem := lbfgsb.Problem{
		N: n, M: m,
		Eval:   eval,
		Stop:   stop,
		Bounds: bounds,
	}
	s, _ := problem.New(logger)

	// Initialize a new workspace.
	w := s.Init()

	// Perform the optimization by fitting the model to the initial guess.
	r := s.Fit(x0, w)

	// Output the optimization result: success status (OK), optimal variables (X), and objective value (F).
	fmt.Printf("Converged: %v, X: %v, F: %v\n", r.OK, r.X, r.F)

}
```

## SLSQP

The SLSQP algorithm is a quasi-Newton optimization algorithm that solves general constrained nonlinear optimization problems. 
It is particularly useful in when constrained optimization requires functional constraints.

This Go implementation follows the structure of the Fortran code from  [SLSQP](https://github.com/jacobwilliams/slsqp).
The license can be found in the repository.

```go
package main

import (
	"fmt"
	"github.com/curioloop/optimizer/slsqp"
)

func main() {

	const n = 2 // Define dimension of problem as n.

	// Define the objective function and its gradient.

	// Objective : x[0]^2 + x[1]^2
	obj := func(x []float64, g []float64) (f float64) {
		if g == nil {
			f = x[0]*x[0] + x[1]*x[1]
        } else {
			g[0], g[1] = 2*x[0], 2*x[1]
		}
		return 
	}

	// Define the constraint functions and its normals.

	// Equality constraint: x[0] + x[1] = 1
	cons1 := func(x []float64, g []float64) (f float64) {
		if g == nil {
			f = x[0] + x[1] - 1
		} else {
			g[0], g[1] = 1, 1
		}
		return 
	}

	// Inequality constraint: x[0] >= 1
	cons2 := func(x []float64, g []float64) (f float64) {
		if g == nil {
			f = x[0] - 1
		} else {
			g[0], g[1] = 1, 0
		}
		return
	}

	// Inequality constraint: x[1] >= 0
	cons3 := func(x []float64, g []float64) (f float64) {
		if g == nil {
			f = x[1]
		} else {
			g[0], g[1] = 0, 1
		}
		return
	}

	// Define termination conditions for the optimizer.
	stop := slsqp.Termination{
		Accuracy:      1e-6,
		MaxIterations: 5,
	}

	// Initial guess for the optimization variables.
	// Here, the initial values for x[0] and x[1] are both set to -0.5 which violates all constraints.
	x0 := []float64{-0.5, -0.5}

	// Initialize the optimizer.
	problem := slsqp.Problem{
		N:       n,
		Object:  obj,
		EqCons:  []slsqp.Evaluation{cons1},
		NeqCons: []slsqp.Evaluation{cons2, cons3},
		Stop:    stop,
	}
	s, _ := problem.New()

	// Initialize a new workspace.
	w := s.Init()

	// Perform the optimization by fitting the model to the initial guess.
	r := s.Fit(x0, w)

	// Output the optimization result: success status (OK), optimal variables (X), and objective value (F).
	fmt.Printf("Converged: %v, X: %v, F: %v\n", r.OK, r.X, r.F)

}

```