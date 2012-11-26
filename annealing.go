package shoehorn

import (
	"math"
)

// func (sh *Shoehorn) Annealing(temp_initial, temp_final, temp_decay float64, temp_epochs int, output_prefix string) {
// 	var (
// 		temp_its, t, o, d                      int
// 		temp, sigma, e_cur, e_try, equilibrium float64
// 		location0, location1, errors           []float64
// 		candidate_function                     func(int, float64, [][]float64) []float64
// 		energy_function                        func([][]float64, [][]float64) float64
// 		tm                                     time.Time
// 	)
// 	// Set candidate and energy functions.
// 	candidate_function = CandidateWormhole
// 	energy_function = TotalEnergy
// 	location0 = make([]float64, sh.nd)
// 	sigma = 1.
// 	temp_its = temp_epochs * sh.no
// 	// Perform simulated annealing.
// 	errors = make([]float64, 0)
// 	e_cur = energy_function(sh.O, sh.L)
// 	for t, temp = 0, temp_initial; temp > temp_final; t++ {
// 		tm = time.Now()
// 		o = rand.Intn(sh.no)
// 		// Calculate try error.
// 		location1 = candidate_function(o, sigma, sh.L)
// 		for d = 0; d < sh.nd; d++ {
// 			location0[d] = sh.L[o][d]
// 			sh.L[o][d] = location1[d]
// 		}
// 		e_try = energy_function(sh.O, sh.L)
// 		// Decide whether to accept the new position or not.
// 		if rand.Float64() < math.Exp(-(e_try-e_cur)/temp) {
// 			e_cur = e_try
// 		} else {
// 			for d = 0; d < sh.nd; d++ {
// 				sh.L[o][d] = location0[d]
// 			}
// 		}
// 		errors = append(errors, e_cur)

// 		if len(errors) == temp_its {
// 			errors = make([]float64, 0)
// 			temp *= temp_decay
// 		}

// 		// if len(errors) > 500 {
// 		// 	errors = errors[len(errors)-500:]
// 		// }
// 		// // Compute thermal equilibrium measure.
// 		// equilibrium = ThermalEquilibrium(errors)
// 		// // Perform equilibrium actions.
// 		// if equilibrium < equilibrium_threshold {
// 		// 	// Reduce temperature.
// 		// 	temp *= temp_decay
// 		// 	// Reset errors.
// 		// 	errors = make([]float64, 0)
// 		// }

// 		// Write locations to file.
// 		if output_prefix != "" {
// 			sh.WriteLocations(fmt.Sprintf("%v.csv", output_prefix))
// 		}
// 		// Report status.
// 		fmt.Printf("Epoch %d: Error=%.6e Temp=%.6e Equilibrium=%.6e (took %v).\n", t, e_cur, temp, equilibrium, time.Now().Sub(tm))
// 	}
// 	// Write locations to file before exiting.
// 	if output_prefix != "" {
// 		sh.WriteLocations(fmt.Sprintf("%v.csv", output_prefix))
// 	}
// }

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

// Thermal equilibrium detector.

func ThermalEquilibrium(errors []float64) (equilibrium float64) {
	var (
		i, n                               int
		mx, my, c0, c1, m1, e0, e1, l0, l1 float64
	)
	n = len(errors)
	// Don't attempt to estimate equilibrium until enough data has been attained.
	if n < 50 {
		return math.NaN()
	}
	// Compute statistics.
	for i = 0; i < n; i++ {
		mx += float64(i)
		my += errors[i]
	}
	mx /= float64(n)
	my /= float64(n)
	// Get parameter for null model.
	c0 = my
	// Get parameters for gradient model.
	top, bot := 0., 0.
	for i = 0; i < n; i++ {
		top += (float64(i) - mx) * (errors[i] - my)
		bot += math.Pow(float64(i)-mx, 2.)
	}
	m1 = top / bot
	c1 = my - (m1 * mx)
	// Estimate errors.
	for i = 0; i < n; i++ {
		e0 += math.Pow(errors[i]-c0, 2.)
		e1 += math.Pow(errors[i]-(c1+m1*float64(i)), 2.)
	}
	e0 = math.Sqrt(e0)
	e1 = math.Sqrt(e1)
	// Get log likelihood under models.
	for i = 0; i < n; i++ {
		l0 += NormalLogPdf(errors[i], c0, e0)
		l1 += NormalLogPdf(errors[i], c1+m1*float64(i), e1)
	}
	// Return equilibrium measure.
	equilibrium = l1 - l0
	return
}

func NormalLogPdf(x, m, s float64) (l float64) {
	var v float64 = s * s
	l = -math.Log(math.Sqrt(2.*math.Pi*v)) - (math.Pow(x-m, 2.) / (2. * v))
	return
}
