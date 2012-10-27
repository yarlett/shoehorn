package shoehorn

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"
)

func (sh *Shoehorn) Annealing(percentile0, percentile1 float64, num_percentiles, its_at_temp int, output_prefix string) {
	var (
		o, d, t, it                       int
		sigma, e_cur, e_try, E, p      float64
		location, A, temps, energy_increases []float64
		wormhole                             bool
		tm                                   time.Time
	)
	// Set parameters for generation of candidate locations.
	wormhole = true
	sigma = 1.
	// Generate a sample of energy increases to get a sense of their distribution.
	energy_increases = make([]float64, 0)
	for len(energy_increases) < 10000 {
		o = rand.Intn(sh.no)
		e_cur = sh.EnergyAt(o, sh.L[o])
		location = sh.Candidate(o, wormhole, sigma)
		e_try = sh.EnergyAt(o, location)
		if e_try > e_cur {
			energy_increases = append(energy_increases, e_try-e_cur)
		}
	}
	// Get temperature schedule based on the distribution of .
	temps = make([]float64, 0)
	for _, p = range sh.GetPercentileSchedule(percentile0, percentile1, num_percentiles) {
		if p == 0.0 {
			temps = append(temps, math.SmallestNonzeroFloat64)
		} else {
			temps = append(temps, Quantile(energy_increases, p))
		}
	}
	energy_increases = nil
	// Perform number of required epochs of learning.
	for t = 0; t < len(temps); t++ {
		// Perform required number of iterations at the current temperature.
		for it = 0; it < its_at_temp; it++ {
			tm, E, A = time.Now(), 0., make([]float64, 2)
			// Decide whether to move each object to a new location.
			for o = 0; o < sh.no; o++ {
				// Get the current error (cannot cache error from last epoch as other objects will have relocated).
				e_cur = sh.EnergyAt(o, sh.L[o])
				// Get the error at the new location.
				location = sh.Candidate(o, wormhole, sigma)
				e_try = sh.EnergyAt(o, location)
				// Decide whether to accept the new position.
				if e_try < e_cur || rand.Float64() < math.Exp((e_cur-e_try)/temps[t]) {
					for d = 0; d < sh.nd; d++ {
						sh.L[o][d] = location[d]
					}
					e_cur = e_try
					A[0] += 1.0
				}
				E += e_cur
				A[1] += 1.0
			}
			E /= float64(sh.no)
			// Report on performance of epoch.
			fmt.Printf("Epoch %d/%d It=%d/%d: E=%.6e T=%.6e P(Acc)=%.6e (took %v).\n", t+1, len(temps), it+1, its_at_temp, E, temps[t], A[0]/A[1], time.Now().Sub(tm))
		}
		// Write locations to file.
		if output_prefix != "" {
			sh.WriteLocations(fmt.Sprintf("%v_%v.csv", output_prefix, t))
		}
	}
}

func Quantile(data []float64, percentile float64) (quantile float64) {
	// Otherwise return the empirical quantile for the specified percentile.
	sort.Float64s(data)
	ix := int(percentile * float64(len(data)))
	if ix < 0 {
		ix = 0
	}
	if ix > len(data)-1 {
		ix = len(data) - 1
	}
	return data[ix]
}

func (sh *Shoehorn) GetPercentileSchedule(percentile0, percentile1 float64, num_percentiles int) (percentiles []float64) {
	var i int
	var alpha float64 = (percentile0 - percentile1) / float64(num_percentiles-1)
	for i = 0; i < num_percentiles; i++ {
		percentiles = append(percentiles, percentile0-float64(i)*alpha)
	}
	return
}

func (sh *Shoehorn) GetTemperatureSchedule(temp0, temp1 float64, num_temps int) (temps []float64) {
	var i int
	var alpha float64
	alpha = math.Pow(temp1/temp0, 1./float64(num_temps-2))
	temps = make([]float64, num_temps)
	for i = 0; i < num_temps; i++ {
		if i == num_temps-1 {
			temps[i] = math.SmallestNonzeroFloat64
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

func (sh *Shoehorn) Candidate(object int, wormhole bool, sigma float64) (location []float64) {
	var (
		d, target_object int
	)
	location = make([]float64, sh.nd)
	if wormhole {
		target_object = rand.Intn(sh.no)
	} else {
		target_object = object
	}
	for d = 0; d < sh.nd; d++ {
		location[d] = sh.L[target_object][d] + (sigma * rand.NormFloat64())
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
	// Energy is sum of squared error.
	for f = 0; f < sh.nf; f++ {
		energy += math.Pow((R[f]/mg)-sh.O[object][f], 2.)
	}
	return
}
