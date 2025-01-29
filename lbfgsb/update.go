// Copyright ©2025 curioloop. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lbfgsb

// Subroutine updateCorrection (matupd)
//
// This subroutine updates matrices Y and WY, and forms the middle matrix in B.
func updateCorrection(loc *iterLoc, spec *iterSpec, ctx *iterCtx) {

	n, m := spec.n, spec.m

	d := ctx.d // s = x𝚗𝚎𝚠 - x𝚘𝚕𝚍
	r := ctx.r // y = g𝚗𝚎𝚠 - g𝚘𝚕𝚍

	if len(r) < len(loc.g) {
		panic("bound check error")
	}

	for i, g := range loc.g {
		r[i] = g - r[i]
	}

	rr := ddot(n, r, 1, r, 1) // yᵀy
	dr := ctx.gd - ctx.gdOld  // sᵀy
	y2 := -ctx.gdOld          // ‖ y ‖₂
	if ctx.stp != one {
		dr *= ctx.stp
		y2 *= ctx.stp
		dscal(n, ctx.stp, ctx.d, 1)
	}

	// skip update when curvature condition sᵀy ≤ ‖ y ‖² × 𝚎𝚙𝚜𝚖𝚌𝚑
	if dr <= spec.epsilon*y2 {
		ctx.totalSkipBFGS++
		ctx.updated = false
		if log := spec.logger; log.enable(LogEval) {
			log.log("Skipping L-BFGS update. dr: %f, y2: %f\n", dr, y2)
		}
		return
	}

	ctx.updated = true
	ctx.updates++

	// Update pointers for matrices S and Y
	if ctx.updates <= m {
		ctx.col = ctx.updates
		ctx.tail = (ctx.head + ctx.updates - 1) % m
	} else {
		ctx.tail = (ctx.tail + 1) % m
		ctx.head = (ctx.head + 1) % m
	}

	// Update matrices S and Y
	ws, wy, sy, ss := ctx.ws, ctx.wy, ctx.sy, ctx.ss

	dcopy(n, d, 1, ws[ctx.tail:], m)
	dcopy(n, r, 1, wy[ctx.tail:], m)

	// Update θ = yᵀy / sᵀy
	ctx.theta = rr / dr

	// Update the middle matrix in B

	// Update the upper triangle of SS and the lower triangle of SY

	// Move old information
	col := ctx.col
	if ctx.updates > m {
		for j := 0; j < col-1; j++ {
			dcopy(col-(j+1), ss[(j+1)*m+(j+1):], 1, ss[j*m+j:], 1) // SS upper triangle
			dcopy(j+1, sy[(j+1)*m+1:], 1, sy[j*m:], 1)             // SY lower triangle
		}
	}

	// Add new information
	ptr := ctx.head
	for j := 0; j < col-1; j++ {
		sy[(col-1)*m+j] = ddot(n, d, 1, wy[ptr:], m) // Last row of SY
		ss[j*m+(col-1)] = ddot(n, ws[ptr:], m, d, 1) // Last column of SS
		ptr = (ptr + 1) % m
	}

	// Update diagonal elements
	sy[(col-1)*m+(col-1)] = dr        // sᵀy
	ss[(col-1)*m+(col-1)] = ctx.dSqrt // sᵀs
	if ctx.stp != one {
		ss[(col-1)*m+(col-1)] *= ctx.stp * ctx.stp
	}

}

// Subroutine formT (formt)
//
// This subroutine computes the matrix T = θSᵀS+LD⁻¹Lᵀ.
// And Cholesky factorize T = JJᵀ with Jᵀ stored in the upper triangular of ctx.wt.
func formT(spec *iterSpec, ctx *iterCtx) (info errInfo) {

	m := spec.m
	col := ctx.col
	theta := ctx.theta

	// Form the upper half of T = θSᵀS + LD⁻¹Lᵀ

	// Store T in the upper triangle of the array wt
	wt := ctx.wt
	ss, sy := ctx.ss, ctx.sy

	if col < 0 || col > len(wt) || col > len(ss) {
		panic("bound check error")
	}

	for j := 0; j < col; j++ {
		wt[j] = theta * ss[j]
	}

	// Matrices D and L could calculate from SᵀY
	//   D = 𝚍𝚒𝚊𝚐 { sᵀy }ᵢ₌₁,...,ₙ
	// Lᵢⱼ = { sᵀy₍ᵢⱼ₎ }ᵢ,ⱼ₌ₖ₋ₘ,...,ₖ₋₁ (i > j)
	for i := 1; i < col; i++ {
		for j := i; j < col; j++ {
			ldl, kk := zero, min(i, j)
			for k := 0; k < kk; k++ {
				ldl += sy[i*m+k] * sy[j*m+k] / sy[k*m+k]
			}
			wt[i*m+j] = ldl + theta*ss[i*m+j]
		}
	}

	// Cholesky factorize T = JJᵀ with Jᵀ stored in the upper triangle of wt
	if dpofa(wt, m, col) != 0 {
		info = errNotPosDefT
	}

	return
}

// Subroutine formK (formk)
//
// This subroutine forms the LELᵀ factorization of the indefinite matrix
//
//	K = [-D - YᵀZZᵀY/θ    Laᵀ - Rzᵀ]   where  E = [-I  0]
//	    [La - Rz          θSᵀAAᵀS  ]              [ 0  I]
//
// The matrix K can be shown to be equal to :
//   - the matrix M⁻¹N occurring in section 5.1 of [1],
//   - the matrix M߫⁻¹M߫ in section 5.3.
//
// wn is a double precision array of dimension 2m x 2m.
//
//	On exit the upper triangle of wn stores the LELᵀ factorization
//	of the 2*col x 2*col indefinite matrix
//	              [-D-YᵀZZᵀY/θ  Laᵀ-Rzᵀ]
//	              [La-Rz        θSᵀAAᵀS]
//
// wn1 is a double precision array of dimension 2m x 2m.
//
//	On entry wn1 stores the lower triangular part of
//	              [YᵀZZᵀY   Laᵀ+Rzᵀ]
//	              [La+Rz    SᵀAAᵀS ]
//	  in the previous iteration.
//	On exit wn1 stores the corresponding updated matrices.
//
//	The purpose of wn1 is just to store these inner products
//	so they can be easily updated and inserted into wn.
func formK(spec *iterSpec, ctx *iterCtx) (info errInfo) {

	n, m := spec.n, spec.m
	col, head := ctx.col, ctx.head

	// matrix stride
	m2 := 2 * m
	col2 := 2 * col

	// 2m x 2m
	wn := ctx.wn
	wn1 := ctx.snd

	ws, wy, sy := ctx.ws, ctx.wy, ctx.sy
	if col < 0 || col > len(wn) || col2 < 0 || col2 > len(wn) {
		panic("bound check error")
	}

	// index[:free] are the indices of free variables
	// index[free:] are the indices of bounds variables
	inx := ctx.index[0]

	// state[:enter] are the variables entering the free set,
	// state[leave:] are the variables leaving he free set.
	inx2 := ctx.index[1]

	// Form the lower triangular part of
	//    WN1 = [ YᵀZZᵀY   Laᵀ + Rzᵀ]
	//          [ La + Rz   SᵀAAᵀS  ]
	// La is the strictly lower triangular part of SᵀAAᵀY
	// Rz is the upper triangular part of SᵀZZᵀY

	if ctx.updated {
		if ctx.updates > m {
			// Shift old parts of WN1
			for jy := 0; jy < m-1; jy++ {
				js := m + jy
				y0, y1 := jy*m2, (jy+1)*m2+1
				dcopy(jy+1, wn1[y1:], 1, wn1[y0:], 1) // YᵀZZᵀY
				s0, s1 := js*m2+m, (js+1)*m2+1+m
				dcopy(jy+1, wn1[s1:], 1, wn1[s0:], 1) // SᵀAAᵀS
				r0, r1 := js*m2, (js+1)*m2+1
				dcopy(m-1, wn1[r1:], 1, wn1[r0:], 1) // La + Rz
			}
		}

		pBeg, pEnd := 0, ctx.free // free variables indices
		dBeg, dEnd := ctx.free, n // active bounds indices

		// Add new rows to blocks (1,1), (2,1), and (2,2)
		iptr := (head + col - 1) % m
		jptr := head

		iy := wn1[(col-1)*m2:]   // last row of YᵀZZᵀY
		is := wn1[(m+col-1)*m2:] // last row of SᵀAAᵀS and La + Rz
		for jy := 0; jy < col; jy++ {
			js := m + jy

			temp1, temp2, temp3 := zero, zero, zero
			for _, k1 := range inx[pBeg:pEnd] { // indices in Z
				temp1 += wy[k1*m+iptr] * wy[k1*m+jptr] // YᵀZZᵀY = YᵀY
			}

			for _, k1 := range inx[dBeg:dEnd] { // indices in A
				temp2 += ws[k1*m+iptr] * ws[k1*m+jptr] // SᵀAAᵀS = SᵀS
				temp3 += ws[k1*m+iptr] * wy[k1*m+jptr] // SᵀAAᵀY = SᵀY
			}

			iy[jy] = temp1 // YᵀZZᵀY
			is[js] = temp2 // SᵀAAᵀS
			is[jy] = temp3 // La
			jptr = (jptr + 1) % m
		}

		// Add new column to block (2,1)
		jptr = (head + col - 1) % m
		iptr = head

		jy := wn1[(m*m2)+col-1:] // last column of La + Rz
		for i := 0; i < col; i++ {
			temp3 := zero
			for _, k1 := range inx[pBeg:pEnd] { // indices in Z
				temp3 += ws[k1*m+iptr] * wy[k1*m+jptr] // SᵀZZᵀY = SᵀY
			}
			jy[i*m2] = temp3 // Rz
			iptr = (iptr + 1) % m
		}
	}

	// Modify the old parts in blocks (1,1) and (2,2) due to changes
	nUpdate := col
	if ctx.updated {
		nUpdate-- // ignore last row and col
	}

	enter := ctx.enter
	leave := ctx.leave

	iptr := head
	for iy := 0; iy < nUpdate; iy++ {
		is := m + iy

		jptr := head
		for jy := 0; jy <= iy; jy++ {
			js := m + jy

			temp1, temp2, temp3, temp4 := zero, zero, zero, zero
			for _, k1 := range inx2[:enter] { // from Z to A
				temp1 += wy[k1*m+iptr] * wy[k1*m+jptr] // YᵀZZᵀY = +YᵀY
				temp2 += ws[k1*m+iptr] * ws[k1*m+jptr] // SᵀAAᵀS = -SᵀS
			}
			for _, k1 := range inx2[leave:n] { // from A to Z
				temp3 += wy[k1*m+iptr] * wy[k1*m+jptr] // YᵀZZᵀY = -YᵀY
				temp4 += ws[k1*m+iptr] * ws[k1*m+jptr] // SᵀAAᵀS = +SᵀS
			}

			wn1[iy*m2+jy] += temp1 - temp3 // YᵀZZᵀY
			wn1[is*m2+js] += temp4 - temp2 // SᵀAAᵀS
			jptr = (jptr + 1) % m
		}
		iptr = (iptr + 1) % m
	}

	// Modify the old parts in block (2,1)
	iptr = head
	for is := m; is < m+nUpdate; is++ {
		jptr := head
		for jy := 0; jy < nUpdate; jy++ {

			temp1, temp3 := zero, zero
			for _, k1 := range inx2[:enter] { // from Z to A
				temp1 += ws[k1*m+iptr] * wy[k1*m+jptr] // SᵀAAᵀY = SᵀY
			}
			for _, k1 := range inx2[leave:n] { // from A to Z
				temp3 += ws[k1*m+iptr] * wy[k1*m+jptr] // SᵀZZᵀY = SᵀY
			}

			if is-m <= jy { // Rz
				wn1[is*m2+jy] += temp1 - temp3
			} else { // La (diagonal elem is zero)
				wn1[is*m2+jy] -= temp1 - temp3
			}
			jptr = (jptr + 1) % m
		}
		iptr = (iptr + 1) % m
	}

	// Form the upper triangle of 2*col x 2*col indefinite matrix
	//        [D+YᵀZZᵀY/θ    -Laᵀ+Rzᵀ]
	//        [-La+Rz        θSᵀAAᵀS ]
	// where
	//        D = 𝚍𝚒𝚊𝚐 { sᵀy }ᵢ₌₁,...,ₙ
	theta := ctx.theta
	for iy := 0; iy < col; iy++ {
		is := col + iy
		is1 := m + iy

		// From WN1 lower triangle to WN upper triangle
		for jy := 0; jy <= iy; jy++ {
			js := col + jy
			js1 := m + jy
			wn[jy*m2+iy] = wn1[iy*m2+jy] / theta   // block (1,1) = (YᵀZZᵀY)ᵀ/θ
			wn[js*m2+is] = wn1[is1*m2+js1] * theta // block (2,2) = θ(SᵀAAᵀS)ᵀ
		}

		// From WN1 block (2,1) to WN block (1,2)
		for jy := 0; jy < iy; jy++ {
			wn[jy*m2+is] = -wn1[is1*m2+jy] // block (2,1) = (-La)ᵀ
		}
		for jy := iy; jy < col; jy++ {
			wn[jy*m2+is] = wn1[is1*m2+jy] // block (2,1) = +Rz
		}

		wn[iy*m2+iy] += sy[iy*m+iy] // += D
	}

	// Form the upper triangle of WN= [  LLᵀ          L⁻¹(-Laᵀ+Rzᵀ)]
	//                                [(-La +Rz)L⁻ᵀ   S'AA'Sθ      ]

	// first Cholesky factor (1,1) block of WN to get LLᵀ
	// with Lᵀ stored in the upper triangle of WN.
	if dpofa(wn, m2, col) != 0 {
		info = errNotPosDef1stK
		return
	}

	// then solving Lx = (-Laᵀ+Rzᵀ) to form L⁻¹(-Laᵀ+Rzᵀ) in the (1,2) block of wn.
	for js := col; js < col2; js++ {
		dtrsl(wn, m2, col, wn[js:], m2, solveUpperT)
	}

	// Form SᵀAAᵀSθ + [L⁻¹(-Laᵀ+Rzᵀ)]ᵀ[L⁻¹(-Laᵀ+Rzᵀ)] in the upper triangle of (2,2) block of wn.
	for is := col; is < col2; is++ {
		for js := is; js < col2; js++ {
			wn[is*m2+js] += ddot(col, wn[is:], m2, wn[js:], m2)
		}
	}

	// Cholesky factorization of (2,2) block of wn.
	if dpofa(wn[col*m2+col:], m2, col) != 0 {
		info = errNotPosDef2ndK
		return
	}

	return
}
