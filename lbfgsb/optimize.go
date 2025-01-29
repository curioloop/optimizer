// Copyright ©2025 curioloop. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lbfgsb

import (
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"time"
)

// LogLevel controls the frequency and type of logger output
type LogLevel int

const (
	// LogNoop no output is generated (level < 0)
	LogNoop LogLevel = -1
	// LogLast print only one line at the last iteration
	LogLast LogLevel = 0
	// LogEval print also f and |proj g| every `level` iterations for any (0 < level < 99)
	LogEval LogLevel = 1
	// LogTrace print details of every iteration except n-vectors
	LogTrace LogLevel = 99
	// LogChange print also the changes of active set and final x
	LogChange LogLevel = 100
	// LogVerbose print details of every iteration including x and g (level > 100)
	LogVerbose LogLevel = 101
)

// Logger handles logging output for the optimizer.
// Note the writers must be thread-safe.
type Logger struct {
	Level LogLevel
	Msg   io.Writer // Writer to output log messages.
	Out   io.Writer // Writer for output data.
}

func (l *Logger) enable(level LogLevel) bool {
	return l.Level >= level
}

func (l *Logger) log(format string, a ...any) {
	if len(a) > 0 {
		_, _ = fmt.Fprintf(l.Msg, format, a...)
	} else {
		_, _ = fmt.Fprint(l.Msg, format)
	}
}

func (l *Logger) out(format string, a ...any) {
	if len(a) > 0 {
		_, _ = fmt.Fprintf(l.Out, format, a...)
	} else {
		_, _ = fmt.Fprint(l.Out, format)
	}
}

// Bound represents the bounds for an optimization variable.
type Bound struct {
	hint         bndHint
	Lower, Upper float64
}

// Evaluation is a function type for evaluating the objective function and gradient.
type Evaluation func(x []float64, g []float64) (f float64)

// Termination specifies the stopping criteria for the optimization algorithm.
type Termination struct {
	// The iteration stop when the number of iteration exceeds limit.
	MaxIterations int
	// The iteration stop when the total number of function and gradient evaluation exceeds limit.
	MaxEvaluations int
	// The iteration stop when the CPU time spent on function and gradient evaluation over quota.
	MaxComputations int64
	// The iteration will stop when the function value satisfied:
	//   (fₖ - fₖ₊₁)/𝚖𝚊𝚡(|fₖ|,|fₖ₊₁|,1) ≤ 𝚏𝚊𝚌𝚝𝚛 × 𝚎𝚙𝚜𝚖𝚌𝚑
	EpsAccuracyFactor float64
	// The iteration will stop when the projected gradient satisfied:
	//   𝚖𝚊𝚡( 𝚙𝚛𝚘𝚓 gᵢ₌₁,...,ₙ ) ≤ 𝚙𝚐𝚝𝚘𝚕
	ProjGradTolerance float64
	// The iteration will stop when the function and gradient value satisfied:
	//   ‖ 𝚙𝚛𝚘𝚓 gₖ ‖∞ / (|fₖ| + 1) ≤ 𝚙𝚍𝚝𝚘𝚕
	GradDescentThreshold float64
}

// Problem specifies the problem for L-BFGS-B optimizer.
type Problem struct {
	N      int         // The problem dimension
	M      int         // The correction number of BFGS
	Eval   Evaluation  // Objective function and gradient
	Stop   Termination // Stop condition
	Bounds []Bound     // Optional bounds
	Search *SearchTol  // Optional line-search config
}

// New creates a new L-BFGS-B optimizer for given problem.
func (p *Problem) New(logger *Logger) (optimizer *Optimizer, err error) {

	if logger == nil {
		logger = new(Logger)
		logger.Level = LogNoop
	}
	if logger.Msg == nil {
		logger.Msg = os.Stdout
	}
	if logger.Out == nil {
		logger.Msg = os.Stderr
	}

	n, m := p.N, p.M
	eval, stop, bounds := p.Eval, p.Stop, p.Bounds

	if bounds == nil {
		bounds = make([]Bound, n)
		for i := range bounds {
			bounds[i].Upper = math.NaN()
			bounds[i].Lower = math.NaN()
		}
	}

	stop.MaxEvaluations = max(stop.MaxEvaluations, 0)
	if stop.MaxEvaluations == 0 {
		stop.MaxEvaluations = math.MaxInt
	}

	stop.MaxComputations = max(stop.MaxComputations, 0)
	if stop.MaxComputations > 0 {
		stop.MaxComputations *= time.Second.Nanoseconds()
	}
	if stop.MaxComputations <= 0 {
		stop.MaxComputations = math.MaxInt64
	}

	switch {
	case n <= 0:
		err = errors.New("problem dimension must greater than 0")
	case m <= 0:
		err = errors.New("correction number must greater than 0")
	case eval == nil:
		err = errors.New("evaluation target is required")
	case stop.MaxIterations <= 0:
		err = errors.New("max iteration must greater than 1")
	case !math.IsNaN(stop.EpsAccuracyFactor) && stop.EpsAccuracyFactor < one:
		err = errors.New("machine epsilon factor must not less than 0")
	case !math.IsNaN(stop.ProjGradTolerance) && stop.ProjGradTolerance < zero:
		err = errors.New("gradient projection tolerance must not less than 0")
	case len(bounds) != n:
		err = errors.New("bounds size must equal to n")
	}

	for k, b := range bounds {
		l, u := !math.IsNaN(b.Lower), !math.IsNaN(b.Upper)
		if l && u && b.Lower > b.Upper {
			err = errors.New(fmt.Sprintf("bound range at %d has no feasible solution", k))
			break
		}
		switch {
		case l && u:
			bounds[k].hint = bndBoth
		case l:
			bounds[k].hint = bndLow
		case u:
			bounds[k].hint = bndUp
		default:
			bounds[k].hint = bndNo
		}
	}

	if err != nil {
		return
	}

	epsilon := math.Nextafter(1, 2) - 1
	optimizer = &Optimizer{
		iterSpec{
			n: n, m: m,
			epsilon: epsilon,
			stop:    stop,
			eval:    eval,
			bounds:  bounds,
			logger:  *logger,
			search:  p.Search,
		},
	}
	return
}

// Optimizer implemented using the L-BFGS-B algorithm.
type Optimizer struct {
	iterSpec
}

// Workspace contains the state and context of the optimization process.
// Given problem dimension n and corrections number m,
// total work space is approximately float64[2×mn + 11×m² + 5×n + 8×m].
type Workspace struct {
	n, m int
	iterCtx
}

// Result contains the final result of the optimization process.
type Result struct {
	OK      bool      // Whether the optimization was converged.
	F       float64   // Final function value.
	X, G    []float64 // Final solution and gradient.
	Summary           // Optimization summary.
}

// Summary contains a summary of the optimization process.
type Summary struct {
	Status  iterTask // Final task status after optimization.
	NumIter int      // Number of iterations performed.
	NumEval int      // Number of function and gradient evaluations performed.
}

// Init allocate the workspace for L-BFGS-B optimizer.
// To avoid race conditions, separate workspaces need to be created for each goroutine.
// But multiple workspaces could share one optimizer.
func (o *Optimizer) Init() *Workspace {
	w := new(Workspace)
	w.n, w.m = o.n, o.m
	w.init(w.n, w.m)
	return w
}

// Fit runs the optimization process using the initial guess x and workspace w.
func (o *Optimizer) Fit(x []float64, w *Workspace) *Result {

	if len(x) != o.n {
		panic("initial x dimension not match spec")
	}

	if w.n != o.n || w.m != o.m {
		panic("workspace dimension not match spec")
	}

	loc := iterLoc{
		x: slices.Repeat(x, 1),
		g: make([]float64, len(x)),
	}

	driver := iterDriver{
		optimizer: o,
		workspace: w,
		location:  &loc,
	}

	res := driver.mainLoop()
	return &Result{
		OK: res&iterConv > 0,
		X:  loc.x, F: loc.f, G: loc.g,
		Summary: Summary{
			Status:  res,
			NumIter: w.iter,
			NumEval: w.totalEval,
		},
	}
}
