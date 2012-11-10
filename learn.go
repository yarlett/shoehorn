package shoehorn

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

func (sh *Shoehorn) LearnRepositioning(epochs int, output_prefix string) {
	var (
		e, o, d            int
		e0, e1             float64
		l0, l1             []float64
		candidate_function func(int, *Shoehorn) []float64
		energy_function    func(*Shoehorn) (e float64)
		tm                 time.Time
	)
	// Initialize.
	l0 = make([]float64, sh.nd)
	l1 = make([]float64, sh.nd)
	// Set candidate and energy functions.
	candidate_function = CandidateHybrid
	energy_function = TotalEnergyAtKL
	sh.NormalizeObjects(1.)
	// Perform learning.
	e0 = energy_function(sh)
	for e = 0; e < epochs; e++ {
		tm = time.Now()
		// Select an object to reposition.
		o = rand.Intn(sh.no)
		// Save the current position.
		for d = 0; d < sh.nd; d++ {
			l0[d] = sh.L[o][d]
		}
		// Generate a new location for o.
		l1 = candidate_function(o, sh)
		// Save the current location and move the object to the new position.
		for d = 0; d < sh.nd; d++ {
			l0[d] = sh.L[o][d]
			sh.L[o][d] = l1[d]
		}
		// Calculate the error at the new position.
		e1 = energy_function(sh)
		// Update information for next round.
		if e1 < e0 {
			e0 = e1
		} else {
			for d = 0; d < sh.nd; d++ {
				sh.L[o][d] = l0[d]
			}
		}
		// Report status.
		fmt.Printf("Epoch %d: Error=%.6e (took %v).\n", e, e0, time.Now().Sub(tm))
		// Write locations to file.
		if output_prefix != "" {
			sh.WriteLocations(fmt.Sprintf("%v_%v.csv", output_prefix, e))
		}
	}
}

func TotalEnergyAtKL(sh *Shoehorn) (energy float64) {
	/*
		Returns the energy of an object when located in a particular position based on Kullback-Leibler divergence.
	*/
	var (
		o, f int
		p, q float64
	)
	// Generate reconstruction information.
	sh.SetNeighbors()
	sh.SetReconstructions()
	// Compute total energy.
	for o = 0; o < sh.no; o++ {
		for f = 0; f < sh.nf; f++ {
			p = sh.O[o][f]
			if p > 0. {
				q = (0.99 * (sh.WP[o][f] / sh.W[o])) + (0.01 * p)
				energy += p * (math.Log(p) - math.Log(q))
			}
		}
	}
	energy /= float64(sh.no)
	return
}

func CandidateHybrid(object int, sh *Shoehorn) (location []float64) {
	var (
		d, o int
	)
	location = make([]float64, sh.nd)
	// Generate a local candidate position 90% of the time.
	if rand.Float64() < .9 {
		for d = 0; d < sh.nd; d++ {
			location[d] = sh.L[object][d] + (1. * rand.NormFloat64())
		}
		// 10% of the time generate a wormhole candidate position.
	} else {
		o = rand.Intn(sh.no)
		for d = 0; d < sh.nd; d++ {
			location[d] = sh.L[o][d] + (1. * rand.NormFloat64())
		}
	}
	return
}