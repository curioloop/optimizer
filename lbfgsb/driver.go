// Copyright ©2025 curioloop. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lbfgsb

import (
	"fmt"
	"math"
)

// iterDriver is the main driver for iterations in an optimization process,
// responsible for managing the flow of the optimization.
type iterDriver struct {
	optimizer *Optimizer
	workspace *Workspace
	location  *iterLoc
}

// nextLocation determines the next iteration state based on the time limit and
// performs function evaluations for the current iteration.
func (d *iterDriver) nextLocation(iter iterTask) iterTask {
	o, w, loc := d.optimizer, d.workspace, d.location
	if w.iterCtx.global.elapsed() >= o.stop.MaxComputations {
		iter = OverTimeLimit
	} else {
		func() {
			defer func() {
				if r := recover(); r != nil {
					iter = HaltEvalPanic
				}
			}()
			loc.f = o.eval(loc.x, loc.g)
			w.totalEval++
		}()
	}
	return iter
}

// newIteration handles the transition to a new iteration, checking for stopping
// conditions like exceeding iteration limits, evaluation limits, or gradient thresholds.
func (d *iterDriver) newIteration(iter iterTask) iterTask {
	o, w, loc := d.optimizer, d.workspace, d.location
	w.iter++
	if w.iter > o.stop.MaxIterations {
		iter = OverIterLimit
	} else if w.totalEval >= o.stop.MaxEvaluations {
		iter = OverEvalLimit
	} else if w.dNorm <= o.stop.GradDescentThreshold*(1.0+math.Abs(loc.f)) {
		iter = OverGradThresh
	}
	return iter
}

// checkConvergence checks if the convergence criteria have been met based on
// the projected gradient norm and the progress in function value reduction.
func (d *iterDriver) checkConvergence(iter iterTask) iterTask {
	o, w, loc := d.optimizer, d.workspace, d.location
	// Compute the infinity norm of the projected (-)gradient
	w.sbgNrm = projGradNorm(loc, &o.iterSpec)
	if w.sbgNrm <= o.stop.ProjGradTolerance {
		iter = ConvGradProgNorm
	} else if w.iter > 0 {
		tolEps := o.epsilon * o.stop.EpsAccuracyFactor
		change := math.Max(math.Abs(w.fOld), math.Max(math.Abs(loc.f), one))
		if w.fOld-loc.f <= tolEps*change {
			iter = ConvEnoughAccuracy
		}
	}
	return iter
}

// mainLoop is the main execution loop of the iteration process, performing
// multiple operations including checking convergence, performing line searches,
// and updating BFGS matrices. It controls the iteration flow.
func (d *iterDriver) mainLoop() (task iterTask) {

	loc := d.location
	spec := &d.optimizer.iterSpec
	ctx := &d.workspace.iterCtx

	log := spec.logger
	wrk := false

	ctx.clear()
	ctx.global.reset()

	d.printInit()
	projInitActive(loc, spec, ctx)

	// Calculate f₀ and g₀
	if task = d.nextLocation(iterLoop); task == iterLoop {
		task = d.checkConvergence(task)
		if log.enable(LogEval) {
			log.log("At iterate %5d    f= %12.5e    |proj g|= %12.5e\n", ctx.iter, loc.f, ctx.sbgNrm)
			log.out(" %4d %4d     -     -   -     -     -        -   %10.3f %10.3f\n", ctx.iter, ctx.totalEval, ctx.sbgNrm, loc.f)
		}
	}

	info := ok
	for task == iterLoop {

		if info != ok {
			info = ok
			ctx.reset()
			if log.enable(LogLast) {
				log.log("Refreshing LBFGS memory and restarting iteration.\n")
			}
		}

		if log.enable(LogTrace) {
			log.log("\n\nITERATION %5d\n", ctx.iter+1)
		}

		if info, wrk = d.searchGCP(); info != ok {
			continue
		}
		if info = d.minimizeSubspace(wrk); info != ok {
			continue
		}
		if info = d.searchOptimalStep(&task); info != ok {
			continue
		}

		// calculate and print out the quantities related to the new X.
		task = d.newIteration(task)
		task = d.checkConvergence(task)

		// Print iteration information
		d.printIter()

		if task == iterLoop {
			info = d.updateBFGS()
		} else if task&ConvEnoughAccuracy > 0 {
			if ctx.numBack >= searchBackSlow {
				info = warnTooManySearch
			}
		}
	}

	d.printExit(task, info)
	return
}

// searchGCP calculates the Generalized Cauchy Point (GCP) for the current
// iteration and updates the corresponding values in the context.
func (d *iterDriver) searchGCP() (info errInfo, wrk bool) {
	loc := d.location
	spec := &d.optimizer.iterSpec
	ctx := &d.workspace.iterCtx

	// skip the search for GCP.
	if !ctx.constrained && ctx.col > 0 {
		dcopy(spec.n, loc.x, 1, ctx.z, 1)
		wrk = ctx.updated
		ctx.seg = 0
	} else {
		// Compute the Generalized Cauchy Point (GCP).
		ctx.shared.reset()
		if info = cauchy(loc, spec, ctx); info == ok {
			// Count the entering and leaving variables for iter > 0;
			// find the index set of free and active variables at the GCP.
			wrk = freeVar(spec, ctx)
			ctx.totalSegGCP += ctx.seg
		}
		ctx.gcpSearchTime += ctx.shared.elapsed()
	}
	if log := spec.logger; log.enable(LogLast) && info != ok {
		log.log("Singular triangular system detected;\n")
	}
	return
}

// minimizeSubspace performs subspace minimization for the current iteration.
// This involves solving the reduced subspace problem and updating the search direction.
func (d *iterDriver) minimizeSubspace(wrk bool) (info errInfo) {
	loc := d.location
	spec := &d.optimizer.iterSpec
	ctx := &d.workspace.iterCtx

	// Subspace minimization.

	// Solve mₖ(x) with direct primal method from GPC xᶜ
	// by find the best which have t free variable
	//
	// only consider points x = xᶜ + Zₖd߮
	//   Zₖ is n x t selection matrix that span the subspace of the free variables at xᶜ
	//   d߮ is t dimension search direction for free variable
	//
	//   minimize   m߮ₖ(d߮) ≡ d߮ᵀr߮ᶜ + ½d߮ᵀB߮ₖr߮ᶜ + γ
	//   subject to lᵢ - xᶜᵢ ≤ d߮ᵢ ≤uᵢ - xᶜᵢ (i ∈ 𝓕)
	//
	//   reduced Hessian B߮ₖ = ZₖᵀBₖZₖ = θI - ZᵀWMWᵀZ
	//   reduced gradient r߮ᶜ = Zₖᵀ(gₖ + Bₖ(xᶜ-xₖ)) = Zᵀ(g + θ(xᶜ-x) - WMc)
	//
	//   the unconstrained solution of the subspace problem is
	//   d߮ᵘ = -B߮ₖ⁻¹r߮ᶜ = r߮ᶜ/θ + ZᵀW(I-MWᵀZZᵀW/θ)⁻¹MWᵀZ/θ²
	//
	//  middle matrix (I-MWᵀZZᵀW/θ)⁻¹M can be written as the inverse of indefinite matrix
	//
	//     K = [-D - YᵀZZᵀY/θ    Laᵀ - Rzᵀ]
	//	       [La - Rz          θSᵀAAᵀS  ]
	//
	// where
	//    La is the lower triangle of SᵀAAᵀS
	//    Rz is the upper triangle of SᵀZZᵀY
	//
	// K can be factorized to LELᵀ where
	//            E = [-I  0]
	//                [ 0  I]
	//

	ctx.word = solutionUnknown

	// If there are no free variables or B = θI, then skip the subspace minimization.
	if ctx.free > 0 && ctx.col > 0 {
		ctx.shared.reset()
		if wrk {
			// K = LELᵀ
			info = formK(spec, ctx)
		}
		if info == ok {
			// r߮ᶜ = -Zᵀ(g + B(xᶜ - xₖ))
			info = reduceGradient(loc, spec, ctx)
		}
		if info == ok {
			// x̂ = xᶜ + d߮⁎
			info = optimalDirection(loc, spec, ctx)
		}
		ctx.minSubspaceTime += ctx.shared.elapsed()
	}

	if log := spec.logger; log.enable(LogLast) && info != ok {
		switch info {
		case errNotPosDef1stK, errNotPosDef2ndK:
			log.log("Nonpositive definiteness in Cholesky factorization in formk;\n")
		default:
			log.log("Singular triangular system detected;\n")
		}
	}
	return
}

// searchOptimalStep calculates the optimal step size (λₖ) for the current iteration,
// using line search techniques to determine the next location in the optimization process.
func (d *iterDriver) searchOptimalStep(task *iterTask) (info errInfo) {

	loc := d.location
	spec := &d.optimizer.iterSpec
	ctx := &d.workspace.iterCtx

	// compute a step length λₖ and set xₖ₊₁ = xₖ + λₖdₖ
	// Generate the search direction dₖ = xᶜ - xₖ
	if x, d, z := loc.x, ctx.d, ctx.z; len(x) != len(d) || len(d) != len(z) {
		panic("bound check error")
	} else {
		for i, x := range x {
			d[i] = z[i] - x
		}
	}

	ctx.shared.reset()
	initLineSearch(loc, spec, ctx)
	loc.save(ctx.t, &ctx.fOld, ctx.r) // Save original x, f, g

	done := false
	for !done {
		info, done = performLineSearch(loc, spec, ctx)
		if info == ok && ctx.numBack < searchBackExit {
			if !done {
				if *task = d.nextLocation(*task); *task&iterStop > 0 {
					break
				} else {
					ctx.numEval++
					ctx.numBack = ctx.numEval - 1
				}
			}
			continue
		}
		if ctx.col == 0 {
			*task = StopAbnormalSearch
			if info == ok {
				info = errLineSearchFailed
			}
			ctx.iter++
		} else {
			info = warnRestartLoop
		}
		break
	}

	if !done {
		// Restore the previous iterate
		loc.load(ctx.t, ctx.fOld, ctx.r)
	}

	if log := spec.logger; log.enable(LogLast) && info != ok {
		switch info {
		case errDerivative:
			log.log("Ascent direction in projection gd = %f\n", ctx.gd)
		case warnRestartLoop:
			log.log("Bad direction in the line search;\n")
		}
	}

	ctx.lineSearchTime += ctx.shared.elapsed()
	return
}

// updateBFGS updates the BFGS correction for the current iteration, updating the
// approximation of the inverse Hessian matrix.
func (d *iterDriver) updateBFGS() (info errInfo) {

	loc := d.location
	spec := &d.optimizer.iterSpec
	ctx := &d.workspace.iterCtx

	updateCorrection(loc, spec, ctx)
	info = formT(spec, ctx)
	if log := spec.logger; log.enable(LogLast) && info != ok {
		log.log("Nonpositive definiteness in Cholesky factorization in formt;\n")
	}

	return
}

// printInit logs the initialization details of the L-BFGS-B optimization process,
// including machine precision, problem dimensions, and initial bounds.
func (d *iterDriver) printInit() {

	loc := d.location
	spec := &d.optimizer.iterSpec

	log := spec.logger

	if log.enable(LogLast) {
		log.log("RUNNING THE L-BFGS-B CODE\n")
		log.log("           * * *\n")
		log.log("Machine precision = %10.3e\n", spec.epsilon)
		log.log("N = %d    M = %d\n", spec.n, spec.m)

		if log.enable(LogEval) {
			log.out("RUNNING THE L-BFGS-B CODE\n\n")
			log.out("Machine precision = %10.3e\n", spec.epsilon)
			log.out("N = %d    M = %d\n", spec.n, spec.m)
			log.out("\n   it   nf   nseg   nact   sub   itls   stepl   tstep   projg      f\n")

			if log.enable(LogVerbose) {
				log.log("\nL  = ")
				for i, b := range spec.bounds {
					log.log("%.2e ", b.Lower)
					if (i+1)%6 == 0 {
						log.log("\n     ")
					}
				}

				log.log("\nX0 = ")
				for i, x := range loc.x {
					log.log("%.2e ", x)
					if (i+1)%6 == 0 {
						log.log("\n     ")
					}
				}

				log.log("\nU  = ")
				for i, b := range spec.bounds {
					log.log("%.2e ", b.Upper)
					if (i+1)%6 == 0 {
						log.log("\n     ")
					}
				}
				log.log("\n")
			}
		}
	}
}

// printIter logs the current iteration details, including the function value,
// gradient norm, and other iteration statistics.
func (d *iterDriver) printIter() {

	loc := d.location
	spec := &d.optimizer.iterSpec
	ctx := &d.workspace.iterCtx

	log := spec.logger

	stpNorm := ctx.stp * ctx.dNorm
	if log.enable(LogTrace) {
		log.log("LINE SEARCH %d times; norm of step = %12.5e\n", ctx.numBack, stpNorm)
		log.log("At iterate %5d    f= %12.5e    |proj g|= %12.5e\n", ctx.iter, loc.f, ctx.sbgNrm)
		var warn string
		switch ctx.task {
		case SearchWarnRoundErr:
			warn = "ROUNDING ERRORS PREVENT PROGRESS"
		case SearchWarnReachEps:
			warn = "XTOL TEST SATISFIED"
		case SearchWarnReachMax:
			warn = "STP = STPMAX"
		case SearchWarnReachMin:
			warn = "STP = STPMIN"
		}
		if warn != "" {
			log.log("WARNING: %v\n", warn)
		}
		if log.enable(LogVerbose) {
			log.log("\n X = ")
			for i := 0; i < spec.n; i++ {
				log.log("%.2e ", loc.x[i])
				if (i+1)%6 == 0 {
					log.log("\n     ")
				}
			}

			log.log("\n G = ")
			for i := 0; i < spec.n; i++ {
				log.log("%.2e ", loc.g[i])
				if (i+1)%6 == 0 {
					log.log("\n     ")
				}
			}
		}
	} else if log.enable(LogEval) {
		if ctx.iter%int(log.Level) == 0 {
			log.log("At iterate %5d    f= %12.5e    |proj g|= %12.5e\n", ctx.iter, loc.f, ctx.sbgNrm)
		}
	}

	if log.enable(LogEval) {
		word := formatWord(ctx.word)
		log.out("%4d %5d %5d %5d %s %4d %7.1f %7.1f %10.3e %10.3e\n",
			ctx.iter, ctx.totalEval, ctx.seg, ctx.active, word, ctx.numBack, ctx.stp, stpNorm, ctx.sbgNrm, loc.f)
	}
}

// printExit logs the final statistics and exit conditions of the optimization process.
func (d *iterDriver) printExit(task iterTask, info errInfo) {

	loc := d.location
	spec := &d.optimizer.iterSpec
	ctx := &d.workspace.iterCtx

	log := spec.logger
	if !log.enable(LogLast) {
		return
	}

	stpNorm := ctx.stp * ctx.dNorm
	time := ctx.global.elapsed()

	log.log("\n           * * *\n")
	log.log("Tit   = total number of iterations\n")
	log.log("Tnf   = total number of function evaluations\n")
	log.log("Tnint = total number of segments explored during Cauchy searches\n")
	log.log("Skip  = number of BFGS updates skipped\n")
	log.log("Nact  = number of active bounds at final generalized Cauchy point\n")
	log.log("Projg = norm of the final projected gradient\n")
	log.log("F     = final function value\n")
	log.log("\n           * * *\n")
	log.log("\n   N      Tit      Tnf   Tnint   Skip   Nact    Projg         F\n")
	log.log("%5d %6d %7d %6d %6d %6d %6.2e %9.5e\n",
		spec.n, ctx.iter, ctx.totalEval, ctx.totalSegGCP, ctx.totalSkipBFGS, ctx.active, ctx.sbgNrm, loc.f)

	if log.enable(LogChange) {
		log.log("\n X =")
		for i := 0; i < spec.n; i++ {
			log.log(" %.2e", loc.x[i])
			if (i+1)%6 == 0 {
				log.log("\n     ")
			}
		}
		log.log("\n")
	}

	if log.enable(LogEval) {
		log.log(" F = %.9e\n", loc.f)
	}

	var msg string
	switch task {
	case ConvGradProgNorm:
		msg = "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL"
	case ConvEnoughAccuracy:
		msg = "CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH"
	case StopAbnormalSearch:
		msg = "ABNORMAL_TERMINATION_IN_LNSRCH"
	case HaltEvalPanic:
		msg = "STOP: CALLBACK REQUESTED HALT"
	case OverIterLimit:
		msg = "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT"
	case OverEvalLimit:
		msg = "STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT"
	case OverTimeLimit:
		msg = "STOP: CPU EXCEEDING THE TIME LIMIT"
	case OverGradThresh:
		msg = "STOP: THE PROJECTED GRADIENT IS SUFFICIENTLY SMALL"
	default:
		msg = "UNKNOWN TASK"
	}
	log.log("\n%s\n", msg)

	if info != ok {
		switch info {
		case errNotPosDef1stK:
			log.log("\n Matrix in 1st Cholesky factorization in formk is not Pos. Def.\n")
		case errNotPosDef2ndK:
			log.log("\n Matrix in 2nd Cholesky factorization in formk is not Pos. Def.\n")
		case errNotPosDefT:
			log.log("\n Matrix in the Cholesky factorization in formt is not Pos. Def.\n")
		case errDerivative:
			log.log("\n Derivative >= 0, backtracking line search impossible.\n")
			log.log("   Previous x, f and g restored.\n")
			log.log(" Possible causes: 1 error in function or gradient evaluation;\n")
			log.log("                  2 rounding errors dominate computation.\n")
		case warnTooManySearch:
			log.log("\n Warning:  more than 10 function and gradient evaluations in the last line search.\n")
			log.log("   Termination may possibly be caused by a bad search direction.\n")
		case errSingularTriangular:
			log.log("\n The triangular system is singular.\n")
		case errLineSearchFailed:
			log.log("\n Line search cannot locate an adequate point after 20 function and gradient evaluations.\n")
			log.log("   Previous x, f and g restored.\n")
			log.log(" Possible causes: 1 error in function or gradient evaluation;\n")
			log.log("                  2 rounding error dominate computation.\n")
		case errLineSearchTol:
			switch ctx.task {
			case SearchErrOverLower:
				msg = "STP < STPMIN"
			case SearchErrOverUpper:
				msg = "STP > STPMAX"
			case SearchErrNegInitG:
				msg = "INITIAL G >= ZERO"
			case SearchErrNegAlpha:
				msg = "FTOL < ZERO"
			case SearchErrNegBeta:
				msg = "GTOL < ZERO"
			case SearchErrNegEps:
				msg = "XTOL < ZERO"
			case SearchErrLower:
				msg = "STPMIN < ZERO"
			case SearchErrUpper:
				msg = "STPMAX < STPMIN"
			}
			log.log("\n Line search setting is invalid: %v \n", msg)
		}
	}

	if log.enable(LogEval) {
		log.log("\n Cauchy                time: %s \n", formatNs(ctx.gcpSearchTime))
		log.log(" Subspace minimization time: %s \n", formatNs(ctx.minSubspaceTime))
		log.log(" Line search           time: %s \n", formatNs(ctx.lineSearchTime))
	}
	log.log("\n Total User time: %s\n", formatNs(time))

	if log.enable(LogEval) {
		if info == errDerivative || info == errLineSearchFailed {
			word := formatWord(ctx.word)
			log.out("\n%4d %5d %5d %5d %s %4d %7.1f %7.1f\n",
				ctx.iter, ctx.totalEval, ctx.seg, ctx.active, word, ctx.numBack, ctx.stp, stpNorm)
		}
	}
}

func formatWord(iword int) string {
	// the ctx of the subspace minimization
	switch iword {
	case solutionWithinBox:
		return "con" // the subspace minimization converged.
	case solutionBeyondBox:
		return "bnd" // the subspace minimization stopped at a bound.
	default:
		return "---"
	}
}

func formatNs(nanoseconds int64) string {
	switch {
	case nanoseconds >= 1e9: // Convert to seconds
		return fmt.Sprintf("%.2f s", float64(nanoseconds)/1e9)
	case nanoseconds >= 1e6: // Convert to milliseconds
		return fmt.Sprintf("%.2f ms", float64(nanoseconds)/1e6)
	case nanoseconds >= 1e3: // Convert to microseconds
		return fmt.Sprintf("%.2f µs", float64(nanoseconds)/1e3)
	default: // Keep in nanoseconds
		return fmt.Sprintf("%.2f ns", float64(nanoseconds))
	}
}
