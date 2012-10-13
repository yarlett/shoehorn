package shoehorn

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

func (sh *Shoehorn) Annealing(temp0, temp1 float64, numepochs int, output_prefix string) {
	var (
		epoch, o, d int
		sigma, temp, temp_decay, e, E float64
		location, errs []float64
		t time.Time
	)
	sigma = 1.
	location = make([]float64, sh.nd)
	errs = make([]float64, sh.no)
	for o = 0; o < sh.no; o++ {
		errs[o] = sh.ErrorAtCos(o, sh.L[o])
	}
	// Determine temperature decay required based on numepochs.
	temp_decay = math.Pow((temp1 / temp0), 1.0/float64(numepochs))
	// Perform number of required epochs of learning.
	for temp, epoch = temp0, 1; temp > temp1; temp *= temp_decay {
		t, E = time.Now(), 0.0
		// Decide whether to move each object to a new location.
		for o = 0; o < sh.no; o++ {
			// Generate the new location.
			for d = 0; d < sh.nd; d++ {
				location[d] = sh.L[o][d] + (sigma * rand.NormFloat64())
			}
			// Get the error at the new location.
			e = sh.ErrorAtCos(o, location)
			// Decide whether to accept the new position.
			if e - errs[o] < temp {
				for d = 0; d < sh.nd; d++ {
					sh.L[o][d] = location[d]
					errs[o] = e
				}
			}
			E += e
		}
		E /= float64(sh.no)
		// Repeat on performance of epoch.
		fmt.Printf("Epoch %d: E=%.6e T=%.6e (took %v).\n", epoch, E, temp, time.Now().Sub(t))
		epoch++
		// Write locations to file.
		if output_prefix != "" {
			sh.WriteLocations(fmt.Sprintf("%v_%v.csv", output_prefix, epoch))
		}
	}
}

func (sh *Shoehorn) ErrorAtCos(object int, location []float64) (cos float64) {
	var (
		o, d, f int
		w, mg float64
		R []float64
	)
	R = make([]float64, sh.nf)
	// Generate reconstruction.
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
	// Compute cos between object and reconstruction.
	for f = 0; f < sh.nf; f++ {
		cos += sh.O[object][f] * R[f]		
		mg += R[f] * R[f]
	}
	cos /= math.Sqrt(mg)
	cos = 1.0 - cos
	return
}