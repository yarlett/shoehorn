package shoehorn

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"time"
)

func (sh *Shoehorn) LearnRepositioning(epochs int, output_prefix string) {
	var (
		e, o, f, d         int
		e0, e1, sigma      float64
		l0, l1, S          []float64
		candidate_function func(int, float64, *Shoehorn) []float64
		energy_function    func([][]float64, []float64, [][]float64) (e float64)
		accepts            []bool
		tm                 time.Time
	)
	// Initialize.
	l0 = make([]float64, sh.nd)
	l1 = make([]float64, sh.nd)
	accepts = make([]bool, 0)
	sigma = 5.
	S = make([]float64, sh.no)
	for o = 0; o < sh.no; o++ {
		for f = 0; f < sh.nf; f++ {
			S[o] += sh.O[o][f]
		}
	}
	// Set candidate and energy functions.
	candidate_function = CandidateAwesome
	energy_function = TotalEnergy
	sh.Rescale(10.)
	// Perform learning.
	e0 = energy_function(sh.O, S, sh.L)
	for e = 0; e < epochs; e++ {
		tm = time.Now()
		// Select an object to reposition.
		o = rand.Intn(sh.no)
		// Save the current position.
		for d = 0; d < sh.nd; d++ {
			l0[d] = sh.L[o][d]
		}
		// Generate a new location for o.
		l1 = candidate_function(o, sigma, sh)
		// Save the current location and move the object to the new position.
		for d = 0; d < sh.nd; d++ {
			l0[d] = sh.L[o][d]
			sh.L[o][d] = l1[d]
		}
		// Calculate the error at the new position.
		e1 = energy_function(sh.O, S, sh.L)
		// Update information for next round.
		if e1 < e0 {
			e0 = e1
			accepts = append(accepts, true)
		} else {
			for d = 0; d < sh.nd; d++ {
				sh.L[o][d] = l0[d]
			}
			accepts = append(accepts, false)
		}
		// Compute rolling accept rate.
		if len(accepts) > 1000 {
			accepts = accepts[len(accepts)-1000:]
		}
		pacc := 0.
		for a := 0; a < len(accepts); a++ {
			if accepts[a] {
				pacc += 1.
			}
		}
		pacc /= float64(len(accepts))
		// Write locations to file.
		if output_prefix != "" {
			sh.WriteLocations(fmt.Sprintf("%v.csv", output_prefix))
		}
		// Report status.
		fmt.Printf("Epoch %d: Error=%.6e P(acc)=%.6e Sigma=%.6e (took %v).\n", e, e0, pacc, sigma, time.Now().Sub(tm))
	}
}

func TotalEnergy(O [][]float64, S []float64, L [][]float64) (energy float64) {
	/*
		Returns the energy of an ensemble of objects.
	*/
	var (
		o, f int
		R [][]float64
	)
	// Generate reconstructions.
	R = SetReconstructions(O, S, L)
	// Compute total energy.
	for o = 0; o < len(O); o++ {
		for f = 0; f < len(O[o]); f++ {
			//energy += math.Pow(O[o][f]-R[o][f], 2.)
			if O[o][f] > 0. {
				energy += math.Abs(O[o][f]-R[o][f]) / O[o][f]
			}
		}
	}
	energy /= float64(len(O))
	return
}

func SetReconstructions(O [][]float64, S []float64, L [][]float64) (R [][]float64) {
	var (
		object  int
		channel chan bool
	)
	// Initialization.
	runtime.GOMAXPROCS(runtime.NumCPU())
	channel = make(chan bool, len(O))
	R = make([][]float64, len(O))
	// Create goroutines to compute reconstruction of each object.
	for object = 0; object < len(O); object++ {
		go SetReconstruction(object, O, S, L, R, channel)
	}
	// Wait for all goroutines to signal completion.
	for object = 0; object < len(O); object++ {
		<-channel
	}
	return
}

func SetReconstruction(object int, O [][]float64, S []float64, L [][]float64, R [][]float64, channel chan bool) {
	/*
		Sets the reconstruction information for the specified object.
	*/
	var (
		o, d, f             int
		w, inc, sumr, scale float64
	)
	R[object] = make([]float64, len(O[object]))
	// Compute the reconstruction information.
	for o = 0; o < len(O); o++ {
		if o != object {
			// Get weight.
			w = 0.
			for d = 0; d < len(L[object]); d++ {
				w += math.Pow(L[object][d]-L[o][d], 2.)
			}
			w = math.Exp(-math.Sqrt(w))
			// Accumulate.
			for f = 0; f < len(O[object]); f++ {
				inc = w * O[o][f]
				R[object][f] += inc
				sumr += inc
			}
		}
	}
	// Normalize reconstruction (so it has same sum of object data).
	scale = S[object] / sumr
	for f = 0; f < len(O[object]); f++ {
		R[object][f] *= scale
	}
	// Signal completion.
	channel <- true
}

func CandidateAwesome(object int, sigma float64, sh *Shoehorn) (location []float64) {
	location = make([]float64, sh.nd)
	o := rand.Intn(sh.no)
	for d := 0; d < sh.nd; d++ {
		location[d] = sh.L[o][d] + (sigma * rand.NormFloat64())
	}
	return
}