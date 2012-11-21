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
		epoch  int
		energy float64
		TM, tm time.Time
	)
	TM = time.Now()
	// Perform learning.
	energy = TotalEnergy(sh.O, sh.L)
	for epoch = 0; epoch < epochs; epoch++ {
		tm = time.Now()
		energy = UpdaterGAT(sh.O, sh.L, energy)
		// Write locations to file.
		if output_prefix != "" {
			sh.WriteLocations(fmt.Sprintf("%v.csv", output_prefix))
		}
		// Report status.
		fmt.Printf("Epoch %d: Energy=%.6e (took %v; %v elapsed).\n", epoch, energy, time.Now().Sub(tm), time.Now().Sub(TM))
	}
}

func (sh *Shoehorn) Foo(O, L [][]float64, output_prefix string) {
	var (
		o1, o2, d                int
		this_energy, best_energy float64
		best_location            []float64
		tm                       time.Time
	)
	for o1 = 0; o1 < len(O); o1++ {
		tm = time.Now()
		best_energy = TotalEnergy(O, L)
		best_location = make([]float64, len(L[o1]))
		for d = 0; d < len(L[o1]); d++ {
			best_location[d] = L[o1][d]
		}
		for o2 = 0; o2 < len(O); o2++ {
			// Relocate o1 to o2's location (plus a tiny bit of noise).
			for d = 0; d < len(L[o1]); d++ {
				L[o1][d] = L[o2][d] + (.1 * rand.NormFloat64())
			}
			this_energy = TotalEnergy(O, L)
			// Save position if it's better.
			if this_energy < best_energy {
				for d = 0; d < len(L[o1]); d++ {
					best_location[d] = L[o1][d]
				}
				best_energy = this_energy
			}
		}
		// Move o1 to the best position.
		for d = 0; d < len(L[o1]); d++ {
			L[o1][d] = best_location[d]
		}
		// Write locations to file.
		if output_prefix != "" {
			sh.WriteLocations(fmt.Sprintf("%v.csv", output_prefix))
		}
		fmt.Printf("Object %v relocated E=%e (took %v).\n", o1, best_energy, time.Now().Sub(tm))
	}
}

func UpdaterGAT(O, L [][]float64, current_energy float64) (energy float64) {
	/*
		Implements a generate-and-test location updater.
	*/
	var (
		o, oo, d       int
		new_energy     float64
		curloc, newloc []float64
	)
	// Randomly select an object to update.
	o = rand.Intn(len(O))
	// Initialize storage for locations.
	curloc = make([]float64, len(L[o]))
	newloc = make([]float64, len(L[o]))
	// Generate a new location for this object.
	oo = rand.Intn(len(O))
	for d = 0; d < len(L[o]); d++ {
		curloc[d] = L[o][d]
		newloc[d] = L[oo][d] + (.1 * rand.NormFloat64())
	}
	// Get energy at new location.
	for d = 0; d < len(L[o]); d++ {
		L[o][d] = newloc[d]
	}
	new_energy = TotalEnergy(O, L)
	// Decide whether to keep new location.
	if new_energy < current_energy {
		energy = new_energy
	} else {
		for d = 0; d < len(L[o]); d++ {
			L[o][d] = curloc[d]
		}
		energy = current_energy
	}
	return
}

func UpdaterGRAD(O, L [][]float64, current_energy float64) (new_energy float64) {
	/*
		Implements a gradient-based location updater.
	*/
	var (
		o, d int
		G       []float64
	)
	// Randomly select an object to update.
	o = rand.Intn(len(O))
	// Perform gradient descent.
	new_energy, G = GetGradient(o, O, L)
	mg := VectorMagnitude(G)
	for d = 0; d < len(G); d++ {
		L[o][d] = L[o][d] - (.1 * G[d] / mg)
	}
	return
}

func TotalEnergy(O [][]float64, L [][]float64) (energy float64) {
	/*
		Returns the energy of an ensemble of objects.
	*/
	var (
		o, f int
		RW   []float64
		R    [][]float64
	)
	// Generate reconstructions.
	R, RW = GetReconstructions(O, L)
	// Compute total energy.
	for o = 0; o < len(O); o++ {
		for f = 0; f < len(O[o]); f++ {
			energy += math.Pow(O[o][f]-(R[o][f]/RW[o]), 2.)
		}
	}
	energy /= float64(len(O))
	return
}

func GetReconstructions(O [][]float64, L [][]float64) (R [][]float64, RW []float64) {
	/*
		Returns the reconstructions for each object in the ensemble.
	*/
	var (
		object  int
		channel chan bool
	)
	// Initialization.
	channel = make(chan bool, len(O))
	R = make([][]float64, len(O))
	RW = make([]float64, len(O))
	// Ensure code runs on multiple cores if available.
	runtime.GOMAXPROCS(runtime.NumCPU())
	// Create goroutines to compute reconstruction of each object.
	for object = 0; object < len(O); object++ {
		go GetReconstruction(object, O, L, R, RW, channel)
	}
	// Wait for all goroutines to signal completion.
	for object = 0; object < len(O); object++ {
		<-channel
	}
	return
}

func GetReconstruction(object int, O [][]float64, L [][]float64, R [][]float64, RW []float64, channel chan bool) {
	/*
		Sets the reconstruction information for the specified object.
	*/
	var (
		no, o, nd, d, nf, f int
		w                   float64
	)
	no = len(O)
	nd = len(L[object])
	nf = len(O[object])
	// Initialize reconstruction information for the object.
	R[object] = make([]float64, nf)
	// Compute the reconstruction information.
	for o = 0; o < no; o++ {
		if o != object {
			// Get weight of o with respect to object.
			w = 0.
			for d = 0; d < nd; d++ {
				w += math.Pow(L[object][d]-L[o][d], 2.)
			}
			w = math.Exp(-math.Sqrt(w))
			// Update sum of weights for the object.
			RW[object] += w
			// Add o's contribution to the reconstruction.
			for f = 0; f < nf; f++ {
				R[object][f] += w * O[o][f]
			}
		}
	}
	// Signal completion.
	channel <- true
}

func GetGradient(object int, O [][]float64, L [][]float64) (E float64, G []float64) {
	var (
		no, nd, nf, o, d, f    int
		g, h, distance, weight, tmp float64
		RW, gprime, hprime     []float64
		R                      [][]float64
	)
	// Initializations.
	no = len(O)
	nd = len(L[object])
	nf = len(O[object])
	G = make([]float64, nd)
	// Get the reconstruction data.
	R, RW = GetReconstructions(O, L)
	// Compute impact of object location on its own reconstruction error.
	for f = 0; f < nf; f++ {
		g = 0.
		h = 0.
		gprime = make([]float64, nd)
		hprime = make([]float64, nd)
		for o = 0; o < no; o++ {
			if o != object {
				// Get weight and distance.
				distance = 0.
				for d = 0; d < nd; d++ {
					distance += math.Pow(L[object][d]-L[o][d], 2.)
				}
				distance = math.Sqrt(distance)
				if distance > 0. {
					weight = math.Exp(-distance)
					// Update gradient information.
					for d = 0; d < nd; d++ {
						// Exp.
						tmp = weight * (L[o][d] - L[object][d]) / distance
						gprime[d] += O[o][f] * tmp
						hprime[d] += tmp
						// // Pow.
						// tmp = (L[o][d] - L[object][d]) / (distance * (1. + distance) * (1. + distance))
						// gprime[d] += O[o][f] * tmp
						// hprime[d] += tmp
					}
				}
			}
		}
		g = R[object][f]
		h = RW[object]
		for d = 0; d < nd; d++ {
			G[d] += 2. * ((g / h) - O[object][f]) * ((gprime[d] * h) - (g * hprime[d])) / (h * h)
		}
	}
	// Compute impact of object location on reconstruction error of other objects.
	for o = 0; o < no; o++ {
		if o != object {
			// Get weight and distance.
			distance = 0.
			for d = 0; d < nd; d++ {
				distance += math.Pow(L[object][d]-L[o][d], 2.)
			}
			distance = math.Sqrt(distance)
			if distance > 0. {
				weight = math.Exp(-distance)
				// Update gradient information.
				for f = 0; f < nf; f++ {
					g = R[o][f]
					h = RW[o]
					for d = 0; d < nd; d++ {
						// Exp.
						tmp = weight * (L[o][d] - L[object][d]) / distance
						gprime[d] = O[object][f] * tmp
						hprime[d] = tmp
						// // Pow.
						// tmp = (L[o][d] - L[object][d]) / (distance * (1. + distance) * (1. + distance))
						// gprime[d] = O[object][f] * tmp
						// hprime[d] = tmp
						// Update gradient.
						G[d] += 2. * ((g / h) - O[o][f]) * ((gprime[d] * h) - (g * hprime[d])) / (h * h)
					}
				}
			}
		}
	}
	// Get current total energy.
	E = 0.
	for o = 0; o < no; o++ {
		for f = 0; f < nf; f++ {
			E += math.Pow(O[o][f]-(R[o][f]/RW[o]), 2.)
		}
	}
	E /= float64(no)
	return
}

func EstimateGradient(object int, O [][]float64, L [][]float64, normalize bool) (G []float64) {
	/*
		Returns an empirical estimate of the gradient of the energy with respect to an object's location.
	*/
	var (
		energy0, energy1, shift float64
	)
	// Initialize.
	G = make([]float64, len(L[object]))
	energy0 = TotalEnergy(O, L)
	shift = 1e-10
	// Estimate gradient for each dimension of location.
	for d := 0; d < len(L[object]); d++ {
		L[object][d] += shift
		energy1 = TotalEnergy(O, L)
		G[d] = (energy1 - energy0) / shift
		L[object][d] -= shift
	}
	// Normalize magnitude of gradient if required.
	if normalize {
		mg := VectorMagnitude(G)
		for d := 0; d < len(G); d++ {
			G[d] /= mg
		}
	}
	return
}
