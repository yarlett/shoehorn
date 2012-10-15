package shoehorn

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"
)

func (sh *Shoehorn) Annealing(sigma float64, num_temps, its_at_temp int, l2 float64, output_prefix string) {
	var (
		epoch, o, d, t, temp_ix, it int
		temp, e, E float64
		location, errs, A, dE []float64
		tm time.Time
	)
	location = make([]float64, sh.nd)
	errs = make([]float64, sh.no)
	for o = 0; o < sh.no; o++ {
		errs[o] = sh.ErrorAtCos(o, sh.L[o], l2)
	}
	// Perform number of required epochs of learning.
	epoch = 1
	for t = 0; t < num_temps; t++ {
		// Get current temperature by taking percentile of change in energy distribution.
		dE = sh.GetEnergyDeltaSamples(sigma, l2, 1000)
		temp_ix = int((1.0 - (float64(t) / float64(num_temps-1))) * float64(len(dE)))
		if temp_ix >= len(dE) {
			temp_ix = len(dE) - 1
		}
		temp = dE[temp_ix]
		// Perform required number of iterations at the current temperature.
		for it = 0; it < its_at_temp; it++ {
			tm, E, A = time.Now(), 0.0, make([]float64, 2)
			// Decide whether to move each object to a new location.
			for o = 0; o < sh.no; o++ {
				// Generate the new location.
				for d = 0; d < sh.nd; d++ {
					location[d] = sh.L[o][d] + (sigma * rand.NormFloat64())
				}
				// Get the error at the new location.
				e = sh.ErrorAtCos(o, location, l2)
				// Decide whether to accept the new position.
				if e - errs[o] < temp {
					for d = 0; d < sh.nd; d++ {
						sh.L[o][d] = location[d]
					}
					errs[o] = e
					A[0] += 1.0
				}
				E += errs[o]
				A[1] += 1.0
			}
			E /= float64(sh.no)
			// Repeat on performance of epoch.
			fmt.Printf("Epoch %d/%d It=%d/%d: E=%.6e T=%.6e P(Acc)=%.6e (took %v).\n", epoch, num_temps, it+1, its_at_temp, E, temp, A[0]/A[1], time.Now().Sub(tm))
		}
		// Write locations to file.
		if output_prefix != "" {
			sh.WriteLocations(fmt.Sprintf("%v_%v.csv", output_prefix, epoch))
		}
		epoch++
	}
}

func (sh *Shoehorn) GetEnergyDeltaSamples(sigma, l2 float64, num_samples int) (dE []float64) {
	var (
		o, d int
		e1, e2 float64
		location []float64
	)
	location = make([]float64, sh.nd)
	// Generate a large sample of change in energies.
	dE = make([]float64, 0)
	for ; len(dE) < num_samples; {
		o = rand.Intn(sh.no)
		e1 = sh.ErrorAtCos(o, sh.L[o], l2)
		for d = 0; d < sh.nd; d++ {
			location[d] = sh.L[o][d] + (sigma * rand.NormFloat64())
		}
		e2 = sh.ErrorAtCos(o, location, l2)
		if e2 > e1 {
			dE = append(dE, e2-e1)
		}
	}
	sort.Sort(sort.Float64Slice(dE))
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

func (sh *Shoehorn) ErrorAtCos(object int, location []float64, l2 float64) (cos float64) {
	var (
		f  int
		mg float64
		R []float64
	)
	// Get reconstruction.
	R = sh.ReconstructionAt(object, location)
	// Compute cos between object and reconstruction.
	for f = 0; f < sh.nf; f++ {
		cos += sh.O[object][f] * R[f]		
		mg += R[f] * R[f]
	}
	cos = (1.0 - (cos / math.Sqrt(mg)))
	cos += l2 * VectorMagnitude(location)
	return
}

func (sh *Shoehorn) ErrorAtRMSE(object int, location []float64) (rmse float64) {
	var (
		f  int
		mg float64
		R []float64
	)
	// Get reconstruction.
	R = sh.ReconstructionAt(object, location)
	// Get magnitude of reconstruction.
	for f = 0; f < sh.nf; f++ {
		mg += R[f] * R[f]
	}
	mg = math.Sqrt(mg)
	// Compute RSE between object and reconstruction.
	for f = 0; f < sh.nf; f++ {
		rmse += math.Pow(sh.O[object][f]-(R[f]/mg), 2.0)
	}
	rmse = math.Sqrt(rmse / float64(sh.nf))
	return
}

func (sh *Shoehorn) ErrorAtKLD(object int, location []float64) (kld float64) {
	var (
		f  int
		s, p, q float64
		R []float64
	)
	// Get reconstruction.
	R = sh.ReconstructionAt(object, location)
	// Get sum of reconstruction.
	for f = 0; f < sh.nf; f++ {
		s += R[f]
	}
	// Compute RSE between object and reconstruction.
	for f = 0; f < sh.nf; f++ {
		p = sh.O[object][f]
		q = (0.1 * p) + (0.9 * (R[f] / s))
		kld += p * math.Log(p/q)
	}
	return kld
}
