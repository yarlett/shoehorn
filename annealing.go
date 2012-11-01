package shoehorn

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

func (sh *Shoehorn) Annealing(temp_initial, temp_final, temp_decay, equilibrium_threshold float64, output_prefix string) {
	var (
		t, o, d                                int
		temp, error, e_cur, e_try, equilibrium float64
		location, errors                       []float64
		candidate_function                     func(int, *Shoehorn) []float64
		energy_function                        func(int, []float64, *Shoehorn) (e float64)
		tm                                     time.Time
	)
	// Set candidate and energy functions.
	candidate_function = CandidateWormhole
	energy_function = EnergyAtKL
	sh.NormalizeObjects(1.)
	// Perform simulated annealing.
	errors = make([]float64, 0)
	for t, temp = 0, temp_initial; temp > temp_final; t++ {
		// Iterate through objects.
		tm = time.Now()
		for o, error = 0, 0.; o < sh.no; o++ {
			// Calculate current and try errors.
			e_cur = energy_function(o, sh.L[o], sh)
			location = candidate_function(o, sh)
			e_try = energy_function(o, location, sh)
			// Decide whether to accept the new position or not.
			if e_try < e_cur || rand.Float64() < math.Exp((e_cur-e_try)/temp) {
				for d = 0; d < sh.nd; d++ {
					sh.L[o][d] = location[d]
				}
				error += e_try
			} else {
				error += e_cur
			}
		}
		error = error / float64(sh.no)
		errors = append(errors, error)
		if len(errors) > 1000 {
			errors = errors[len(errors)-1000:]
		}
		// Compute thermal equilibrium measure.
		equilibrium = ThermalEquilibrium(errors)
		// Perform equilibrium actions.
		if equilibrium < equilibrium_threshold {
			// Reduce temperature.
			temp *= temp_decay
			// Reset errors.
			errors = make([]float64, 0)
			// Write locations to file.
			if output_prefix != "" {
				sh.WriteLocations(fmt.Sprintf("%v_%v.csv", output_prefix, t))
			}
		}
		// Report status.
		fmt.Printf("Epoch %d: Error=%.6e Temp=%.6e Equilibrium=%.6e (took %v).\n", t, error, temp, equilibrium, time.Now().Sub(tm))
	}
	// Write locations to file before exiting.
	if output_prefix != "" {
		sh.WriteLocations(fmt.Sprintf("%v_%v.csv", output_prefix, t))
	}
}

func (sh *Shoehorn) ReconstructionAt(object int, location []float64) (R []float64, W float64) {
	var (
		o, d, f int
		w       float64
	)
	// Generate reconstruction.
	R = make([]float64, sh.nf)
	for o = 0; o < sh.no; o++ {
		if o != object {
			// Calculate weight.
			w = 0.
			for d = 0; d < sh.nd; d++ {
				w += math.Pow(location[d]-sh.L[o][d], 2.)
			}
			w = math.Exp(-math.Sqrt(w))
			W += w
			// Accumulate neighbor's distributional information.
			for f = 0; f < sh.nf; f++ {
				R[f] += w * sh.O[o][f]
			}
		}
	}
	return
}

// Candidate functions.

func Candidate(object int, sh *Shoehorn) (location []float64) {
	location = make([]float64, sh.nd)
	for d := 0; d < sh.nd; d++ {
		location[d] = sh.L[object][d] + (1. * rand.NormFloat64())
	}
	return
}

func CandidateWormhole(object int, sh *Shoehorn) (location []float64) {
	var (
		d, target_object int
	)
	location = make([]float64, sh.nd)
	target_object = rand.Intn(sh.no)
	for d = 0; d < sh.nd; d++ {
		location[d] = sh.L[target_object][d] + (1. * rand.NormFloat64())
	}
	return
}

// Energy functions.

func EnergyAtSumSquared(object int, location []float64, sh *Shoehorn) (energy float64) {
	/*
		Returns the energy of an object when located in a particular position based on squared error function.
	*/
	var (
		f  int
		mg float64
		R  []float64
	)
	// Get reconstruction.
	R, _ = sh.ReconstructionAt(object, location)
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

func EnergyAtKL(object int, location []float64, sh *Shoehorn) (energy float64) {
	/*
		Returns the energy of an object when located in a particular position based on Kullback-Leibler divergence.
	*/
	var (
		f, d    int
		W, p, q float64
		R       []float64
	)
	// Get reconstruction.
	R, W = sh.ReconstructionAt(object, location)
	// Energy is sum of squared error.
	for f = 0; f < sh.nf; f++ {
		p = sh.O[object][f]
		if p > 0. {
			q = (0.99 * (R[f] / W)) + (0.01 * p)
			energy += p * (math.Log(p) - math.Log(q))
		}
	}
	// Add L2 punishment.
	for d = 0; d < sh.nd; d++ {
		energy += 0.001 * location[d] * location[d]
	}
	return
}

// Thermal equilibrium detector.

func ThermalEquilibrium(E []float64) (equilibrium float64) {
	var (
		i                          int
		c0, e0, l0, c1, m1, e1, l1 float64
	)
	// Estimate intercept of null model.
	for i = 0; i < len(E); i++ {
		c0 += E[i]
	}
	c0 /= float64(len(E))
	// Estimate error under null model.
	for i = 0; i < len(E); i++ {
		e0 += math.Pow(E[i]-c0, 2.)
	}
	e0 = math.Sqrt(e0 / float64(len(E)))
	// Get log likelihood under null model.
	for i = 0; i < len(E); i++ {
		l0 += NormalLogPdf(E[i], c0, e0)
	}
	// Get means of data.
	mx := float64(len(E)) / 2.
	my := c0
	// Estimate gradient of positive model.
	top, bot := 0., 0.
	for i = 0; i < len(E); i++ {
		top += (float64(i) - mx) * (E[i] - my)
		bot += math.Pow(float64(i)-mx, 2.)
	}
	m1 = top / bot
	// Estimate intercept of positive model.
	c1 = my - m1*mx
	// Estimate error under positive model.
	for i = 0; i < len(E); i++ {
		e1 += math.Pow(E[i]-(c1+m1*float64(i)), 2.)
	}
	e1 = math.Sqrt(e1 / float64(len(E)))
	// Get log likelihood under positive model.
	for i = 0; i < len(E); i++ {
		l1 += NormalLogPdf(E[i], c1+m1*float64(i), e1)
	}
	// Return equilibrium measure.
	equilibrium = l1 / l0
	return
}

func NormalLogPdf(x, m, s float64) (l float64) {
	var v float64 = s * s
	l = -math.Log(math.Sqrt(2.*math.Pi*v)) - (math.Pow(x-m, 2.) / (2. * v))
	return
}
