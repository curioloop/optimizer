// Copyright ©2025 curioloop. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lbfgsb

import (
	"fmt"
	"slices"
	"strings"
	"testing"
)

func TestShiftWN1(t *testing.T) {

	// 2m x 2m
	// [ Y ? ]
	// [ R S ]

	m := 5
	m2 := m * 2
	wn := []float64{
		1, 0, 0, 0, 0 /**/, 1, 1, 1, 1, 1,
		1, 2, 0, 0, 0 /**/, 2, 2, 2, 2, 2,
		1, 2, 3, 0, 0 /**/, 3, 3, 3, 3, 3,
		1, 2, 3, 4, 0 /**/, 4, 4, 4, 4, 4,
		1, 2, 3, 4, 5 /**/, 5, 5, 5, 5, 5,
		/*************************************/
		1, 1, 1, 1, 1 /**/, 1, 0, 0, 0, 0,
		0, 2, 2, 2, 2 /**/, 1, 2, 0, 0, 0,
		0, 0, 3, 3, 3 /**/, 1, 2, 3, 0, 0,
		0, 0, 0, 4, 4 /**/, 1, 2, 3, 4, 0,
		0, 0, 0, 0, 5 /**/, 1, 2, 3, 4, 5,
	}

	wn1 := slices.Repeat(wn, 1)
	for jy := 0; jy < m-1; jy++ {
		js := m + jy
		jy0, jy1 := jy*m2+jy, (jy+1)*m2+(jy+1)
		js0, js1 := js*m2+js, (js+1)*m2+(js+1)
		dcopy((m-1)-jy, wn1[jy1:], m2, wn1[jy0:], m2)
		dcopy((m-1)-jy, wn1[js1:], m2, wn1[js0:], m2)
		dcopy(m-1, wn1[(m+1)*m2+(jy+1):], m2, wn1[m*m2+jy:], m2)
	}
	fmt.Println(formatMat(m2, m2, wn1))

	wn2 := slices.Repeat(wn, 1)
	for jy := 0; jy < m-1; jy++ {
		js := m + jy
		y0, y1 := jy*m2, (jy+1)*m2+1
		copy(wn2[y0:y0+jy+1], wn2[y1:y1+jy+1])
		s0, s1 := js*m2+m, (js+1)*m2+1+m
		copy(wn2[s0:s0+jy+1], wn2[s1:s1+jy+1])
		r0, r1 := js*m2, (js+1)*m2+1
		copy(wn2[r0:r0+m-1], wn2[r1:r1+m-1])
	}
	fmt.Println(formatMat(m2, m2, wn1))

	if !slices.Equal(wn1, wn2) {
		t.Fatalf("shift WN1 failed")
	}

}

//func TestCopyWN(t *testing.T) {
//	m := 5
//	m2 := m * 2
//
//	// [ Y ? ]
//	// [ R S ]
//	wn1 := []float64{
//		1, 0, 0, 0, 0 /**/, 8, 8, 8, 8, 8,
//		1, 2, 0, 0, 0 /**/, 8, 8, 8, 8, 8,
//		1, 2, 3, 0, 0 /**/, 8, 8, 8, 8, 8,
//		1, 2, 3, 4, 0 /**/, 8, 8, 8, 8, 8,
//		1, 2, 3, 4, 5 /**/, 8, 8, 8, 8, 8,
//		/*************************************/
//		1, 1, 1, 1, 1 /**/, 1, 0, 0, 0, 0,
//		2, 2, 2, 2, 2 /**/, 1, 2, 0, 0, 0,
//		3, 3, 3, 3, 3 /**/, 1, 2, 3, 0, 0,
//		4, 4, 4, 4, 4 /**/, 1, 2, 3, 4, 0,
//		5, 5, 5, 5, 5 /**/, 1, 2, 3, 4, 5,
//	}
//
//	theta := 1.
//	sy := make([]float64, m*m)
//	for i := 0; i < m; i++ {
//		sy[i*m+i] = .5
//	}
//
//	// 2m x 2m
//	// [ Y/θ    ? ]
//	// [ -L+R  θS ]
//	for col := 0; col <= m; col++ {
//		wn := make([]float64, m2*m2)
//		for iy := 0; iy < col; iy++ {
//			is := col + iy
//			is1 := m + iy
//
//			for jy := 0; jy <= iy; jy++ {
//				js := col + jy
//				js1 := m + jy
//				wn[jy*m2+iy] = wn1[iy*m2+jy] / theta   // (Y)ᵀ/θ
//				wn[js*m2+is] = wn1[is1*m2+js1] * theta // θ(Sᵀ)
//			}
//
//			for jy := 0; jy < iy; jy++ {
//				wn[jy*m2+is] = -wn1[is1*m2+jy]
//			}
//
//			for jy := iy; jy < col; jy++ {
//				wn[jy*m2+is] = wn1[is1*m2+jy]
//			}
//
//			wn[iy*m2+iy] += sy[iy*m+iy]
//		}
//		fmt.Println(formatMat(m2, m2, wn))
//	}
//
//}

func TestMoveBFGS(t *testing.T) {

	m := 5
	col := m
	r := []float64{
		11, 12, 13, 14, 15,
		21, 22, 23, 24, 25,
		31, 32, 33, 34, 35,
		41, 42, 43, 44, 45,
		51, 52, 53, 54, 55,
	}

	ss1 := slices.Repeat(r, 1)
	sy1 := slices.Repeat(r, 1)
	// Move old information
	for j := 0; j < col-1; j++ {
		dcopy(j+1, ss1[(j+1)+m:], m, ss1[j:], m)               // SS upper triangle
		dcopy(col-j-1, sy1[(j+1)*m+(j+1):], m, sy1[j*m+j:], m) // SY lower triangle
	}

	//fmt.Println("------------------------------------------------")
	//fmt.Println(formatMat(m, m, ss1))
	//fmt.Println("------------------------------------------------")
	//fmt.Println(formatMat(m, m, sy1))

	ss2 := slices.Repeat(r, 1)
	sy2 := slices.Repeat(r, 1)
	// Move old information
	for j := 0; j < col-1; j++ {
		dcopy(col-(j+1), ss2[(j+1)*m+(j+1):], 1, ss2[j*m+j:], 1) // SS upper triangle
		dcopy(j+1, sy2[(j+1)*m+1:], 1, sy2[j*m:], 1)             // SY lower triangle
	}

	//fmt.Println("------------------------------------------------")
	//fmt.Println(formatMat(m, m, ss2))
	//fmt.Println("------------------------------------------------")
	//fmt.Println(formatMat(m, m, sy2))

	if !slices.Equal(ss1, ss2) {
		t.Fatalf("shift SS failed")
	}

	if !slices.Equal(sy2, sy2) {
		t.Fatalf("shift SY failed")
	}

}

func formatMat(rows, cols int, data []float64) string {
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

			sb.WriteString(fmt.Sprintf(" %g", data[i*rows+j]))
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
