package shoehorn

import (
	"math/rand"
)

// Candidate functions.

func CandidateLocal(object int, sigma float64, L [][]float64) (location []float64) {
	location = make([]float64, len(L[object]))
	for d := 0; d < len(L[object]); d++ {
		location[d] = L[object][d] + (sigma * rand.NormFloat64())
	}
	return
}

func CandidateWormhole(object int, sigma float64, L [][]float64) (location []float64) {
	var (
		d, target_object int
	)
	location = make([]float64, len(L[object]))
	target_object = rand.Intn(len(L))
	for d = 0; d < len(L[object]); d++ {
		location[d] = L[target_object][d] + (sigma * rand.NormFloat64())
	}
	return
}

func CandidateHybrid(object int, sigma float64, L [][]float64) (location []float64) {
	if rand.Float64() < .5 {
		location = CandidateLocal(object, sigma, L)
	} else {
		location = CandidateWormhole(object, sigma, L)
	}
	return
}