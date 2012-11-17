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
		energy = UpdaterGRAD(sh.O, sh.L, energy, sh)
		// Write locations to file.
		if output_prefix != "" {
			sh.WriteLocations(fmt.Sprintf("%v.csv", output_prefix))
		}
		// Report status.
		fmt.Printf("Epoch %d: Energy=%.6e (took %v; %v elapsed).\n", epoch, energy, time.Now().Sub(tm), time.Now().Sub(TM))
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
		newloc[d] = L[oo][d] + (2. * rand.NormFloat64())
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

func UpdaterGRAD(O, L [][]float64, current_energy float64, sh *Shoehorn) (new_energy float64) {
	/*
		Implements a gradient-based location updater.
	*/
	var (
		o, oo, d int
		curloc, G    []float64
	)
	// Randomly select an object to update.
	o = rand.Intn(len(O))
	// Try to wormhole the object on a small number of trials.
	if rand.Float64() < .1 {
		curloc = make([]float64, len(L[o]))
		oo = rand.Intn(len(O))
		for d = 0; d < len(L[o]); d++ {
			curloc[d] = L[o][d]
			L[o][d] = L[oo][d] + (.1 * rand.NormFloat64())
		}
		new_energy = TotalEnergy(O, L)
		if new_energy >= current_energy {
			for d = 0; d < len(L[o]); d++ {
				L[o][d] = curloc[d]
			}
			new_energy = current_energy			
		}
		return
	}
	// Perform gradient descent.
	new_energy, G = GetGradient(o, O, L)
	mg := VectorMagnitude(G)
	for d = 0; d < len(G); d++ {
		L[o][d] = L[o][d] - (.5 * G[d] / mg)
	}
	// // Perform line search.
	// stepsize = 2.
	// for t = 0; t < 10; t++ {
	// 	// Set location.
	// 	for d = 0; d < len(L[o]); d++ {
	// 		L[o][d] = curloc[d] - (stepsize * G[d])
	// 	}
	// 	// Get new energy.
	// 	new_energy = TotalEnergy(O, L)
	// 	// Decide whether to keep new location.
	// 	if new_energy < current_energy {
	// 		break
	// 	} else {
	// 		stepsize *= .5
	// 	}
	// }
	// // Stay at current location if energy hasn't been reduced.
	// if new_energy > current_energy {
	// 	for d = 0; d < len(L[o]); d++ {
	// 		L[o][d] = curloc[d]
	// 	}
	// 	new_energy = current_energy
	// }
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
		no, nd, nf, o, d, f                      int
		p, q, tmp1, tmp2, tmp3, weight, distance float64
		RW, T1, T2                               []float64
		R                                        [][]float64
	)
	// Initializations.
	no = len(O)
	nd = len(L[object])
	nf = len(O[object])
	T1 = make([]float64, nd)
	T2 = make([]float64, nd)
	G = make([]float64, nd)
	// Get the reconstruction data.
	R, RW = GetReconstructions(O, L)
	// Compute impact of object position on its own reconstruction error.
	for f = 0; f < nf; f++ {
		p = O[object][f]
		q = R[object][f] / RW[object]
		// Calculate the gradient terms, T1 and T2.
		for d = 0; d < nd; d++ {
			T1[d], T2[d] = 0., 0.
		}
		for o = 0; o < no; o++ {
			if o != object {
				// Get weight and distance.
				distance = 0.
				for d = 0; d < nd; d++ {
					distance += math.Pow(L[object][d]-L[o][d], 2.)
				}
				distance = math.Sqrt(distance)
				weight = math.Exp(-distance)
				// Update gradient terms.
				tmp1 = weight / distance
				for d = 0; d < nd; d++ {
					tmp2 = tmp1 * (L[o][d] - L[object][d])
					T1[d] += tmp2 * O[o][f]
					T2[d] += tmp2
				}
			}
		}
		// Update gradient information.
		tmp1 = 2. * (q - p)
		for d = 0; d < nd; d++ {
			G[d] += tmp1 * (((T1[d] * RW[object]) - (R[object][f] * T2[d])) / (RW[object] * RW[object]))
		}
	}
	// Compute impact of object position on reconstruction error of other objects.
	for o = 0; o < no; o++ {
		if o != object {
			// Calculate distance and weight between current object and object being reconstructed.
			distance = 0.
			for d = 0; d < nd; d++ {
				distance += math.Pow(L[object][d]-L[o][d], 2.)
			}
			distance = math.Sqrt(distance)
			weight = math.Exp(-distance)
			tmp1 = weight / distance
			// Iterate over features of object getting reconstructed.
			for f = 0; f < nf; f++ {
				p = O[o][f]
				q = R[o][f] / RW[o]
				// Update gradient information.
				tmp2 = 2. * (q - p)
				for d = 0; d < nd; d++ {
					tmp3 = tmp1 * (L[o][d] - L[object][d])
					G[d] += tmp2 * (((RW[o] * tmp3 * O[object][f]) - (R[o][f] * tmp3)) / (RW[o] * RW[o]))
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
