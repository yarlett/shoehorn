package shoehorn

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

func (sh *Shoehorn) Annealing(temp0, temp1 float64, num_temps, its_at_temp int, sigma float64, output_prefix string) {
	var (
		o, d, t, it        int
		e1, e2, E          float64
		location, A, temps []float64
		tm                 time.Time
	)
	// Get temperature schedule.
	temps = sh.GetTemperatureSchedule(temp0, temp1, num_temps)
	// Perform number of required epochs of learning.
	for t = 0; t < len(temps); t++ {
		// Perform required number of iterations at the current temperature.
		for it = 0; it < its_at_temp; it++ {
			tm, E, A = time.Now(), 0.0, make([]float64, 2)
			// Decide whether to move each object to a new location.
			for o = 0; o < sh.no; o++ {
				// Get the current error (cannot cache error from last epoch as other objects will have relocated).
				e1 = sh.EnergyAt(o, sh.L[o])
				// Get the error at the new location.
				location = sh.Candidate(o, sigma)
				e2 = sh.EnergyAt(o, location)
				// Decide whether to accept the new position.
				if rand.Float64() < math.Exp((e1-e2)/temps[t]) {
					for d = 0; d < sh.nd; d++ {
						sh.L[o][d] = location[d]
					}
					e1 = e2
					A[0] += 1.0
				}
				E += e1
				A[1] += 1.0
			}
			E /= float64(sh.no)
			// Repeat on performance of epoch.
			fmt.Printf("Epoch %d/%d It=%d/%d: E=%.6e T=%.6e P(Acc)=%.6e (took %v).\n", t+1, num_temps, it+1, its_at_temp, E, temps[t], A[0]/A[1], time.Now().Sub(tm))
		}
		// Write locations to file.
		if output_prefix != "" {
			sh.WriteLocations(fmt.Sprintf("%v_%v.csv", output_prefix, t))
		}
	}
}

func (sh *Shoehorn) GetTemperatureSchedule(temp0, temp1 float64, num_temps int) (temps []float64) {
	var i int
	var alpha float64
	alpha = math.Pow(temp1/temp0, 1./float64(num_temps-2))
	temps = make([]float64, num_temps)
	for i = 0; i < num_temps; i++ {
		if i == num_temps-1 {
			temps[i] = 0.
		} else {
			temps[i] = temp0 * math.Pow(alpha, float64(i))
		}
	}
	return
}

func (sh *Shoehorn) ReconstructionAt(object int, location []float64) (R []float64) {
	var (
		o, d, f int
		w       float64
	)
	// Generate reconstruction.
	R = make([]float64, sh.nf)
	for o = 0; o < sh.no; o++ {
		if o != object {
			// Calculate weight.
			w = 0.0
			for d = 0; d < sh.nd; d++ {
				w += math.Pow(location[d]-sh.L[o][d], 2.0)
			}
			w = math.Exp(-math.Sqrt(w))
			// Accumulate neighbor's distributional information.
			for f = 0; f < sh.nf; f++ {
				R[f] += w * sh.O[o][f]
			}
		}
	}
	return
}

func (sh *Shoehorn) Candidate(object int, sigma float64) (location []float64) {
	var (
		d, o int
	)
	location = make([]float64, sh.nd)
	o = rand.Intn(sh.no)
	for d = 0; d < sh.nd; d++ {
		location[d] = sh.L[o][d] + (sigma * rand.NormFloat64())
	}
	return
}

func (sh *Shoehorn) EnergyAt(object int, location []float64) (energy float64) {
	/*
		Returns the energy of an object when located in a particular position.
	*/
	var (
		f  int
		mg float64
		R  []float64
	)
	// Get reconstruction.
	R = sh.ReconstructionAt(object, location)
	// Get magnitude of reconstruction.
	for f = 0; f < sh.nf; f++ {
		mg += R[f] * R[f]
	}
	mg = math.Sqrt(mg)
	// Energy is squared error.
	for f = 0; f < sh.nf; f++ {
		energy += math.Pow((R[f]/mg)-sh.O[object][f], 2.)
	}
	return
}
