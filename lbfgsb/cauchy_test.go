// Copyright ©2025 curioloop. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lbfgsb

import (
	"math/rand/v2"
	"slices"
	"testing"
)

func TestHpsolb(tt *testing.T) {
	for k := 1; k < 1000; k++ {
		t := make([]float64, k)
		order := make([]int, k)
		for i := 0; i < k; i++ {
			t[i] = float64(i)
			order[i] = i
		}
		rand.Shuffle(k, func(i, j int) {
			t[i], t[j] = t[j], t[i]
			order[i], order[j] = order[j], order[i]
		})
		for n := k; n > 1; n-- {
			heapSortOut(n, t, order, n < k)
			heap, heapOrder := t[:n-1], order[:n-1]
			if slices.Min(heap) != heap[0] || slices.Min(heapOrder) != heapOrder[0] {
				tt.Fatalf("heapsort test failed %v %v", n, t)
			}
			if t[n-1] > heap[0] {
				tt.Fatalf("heapsort test failed %v %v", n, t)
			}
		}
		finalDescending := slices.IsSortedFunc(t, func(a, b float64) int {
			return int(b - a)
		})
		if !finalDescending {
			tt.Fatalf("heapsort test failed %v", t)
		}
	}
}
