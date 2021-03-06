package shoehorn

import (
	"fmt"
	"math"
	"math/rand"
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
	energy = sh.TotalEnergy()
	for epoch = 0; epoch < epochs; epoch++ {
		tm = time.Now()
		energy = GenerateAndTestUpdater(energy, sh)
		// Write locations to file.
		if epoch % 100 == 0 {
			// Write locations to CSV.
			if output_prefix != "" {
				sh.WriteLocations(fmt.Sprintf("%v.csv", output_prefix))
			}
			// Report status.
			fmt.Printf("Epoch %d: Energy=%.6e (took %v; %v elapsed).\n", epoch, energy, time.Now().Sub(tm), time.Now().Sub(TM))
		}
	}
}

func GenerateAndTestUpdater(current_energy float64, sh *Shoehorn) (energy float64) {
	/*
		Implements a generate-and-test location updater.
	*/
	var (
		o, d              int
		new_energy, sigma float64
		curloc, newloc    []float64
	)
	sigma = 1.
	// Randomly select an object to update.
	o = rand.Intn(sh.no)
	// Save the current location of the object.
	curloc = make([]float64, sh.no)
	for d = 0; d < sh.nd; d++ {
		curloc[d] = sh.L[o][d]
	}
	// Generate a new location for the object.
	newloc = CandidateHybrid(o, sigma, sh.L)
	new_energy = sh.TotalEnergyWithRelocation(o, newloc)
	// Decide whether to keep new location.
	if new_energy < current_energy {
		energy = new_energy
	} else {
		energy = sh.TotalEnergyWithRelocation(o, curloc)
	}
	return
}

func (sh *Shoehorn) TotalEnergyWithRelocation(object int, location []float64) (E float64) {
	/*
		Relocates the specified object to the specified location, efficiently updates the internal representation of distances, weights and the object reconstructions, and returns the new global error.
	*/
	var (
		o, d, f                  int
		new_distance, new_weight float64
		old_weights              []float64
	)
	// Update location of object.
	for d = 0; d < sh.nd; d++ {
		sh.L[object][d] = location[d]
	}
	// Take a copy of the old weights involving the object, while setting the new distances and weights.
	old_weights = make([]float64, sh.no)
	for o = 0; o < sh.no; o++ {
		// Store the old weights involving the object.
		old_weights[o] = sh.NW[object][o]
		// Compute the new distance and weight for the object.
		new_distance = 0.
		for d = 0; d < sh.nd; d++ {
			new_distance += math.Pow(sh.L[object][d]-sh.L[o][d], 2.)
		}
		new_distance = math.Sqrt(new_distance)
		new_weight = math.Exp(-new_distance)
		// Set these distances and weights in the internal representation.
		sh.ND[object][o] = new_distance
		sh.ND[o][object] = new_distance
		sh.NW[object][o] = new_weight
		sh.NW[o][object] = new_weight
	}
	// Zero W and WP information for the object.
	sh.W[object] = 0
	for f = 0; f < sh.nf; f++ {
		sh.WP[object][f] = 0
	}
	// Iterate over other objects, o.
	for o = 0; o < sh.no; o++ {
		if o != object {
			// Update W for o.
			sh.W[o] += sh.NW[object][o] - old_weights[o]
			for f = 0; f < sh.nf; f++ {
				sh.WP[o][f] += (sh.NW[object][o] - old_weights[o]) * sh.O[object][f] // Update WP for o.
				sh.WP[object][f] += sh.NW[object][o] * sh.O[o][f] // Update WP for object.
			}
			// Update W for object.
			sh.W[object] += sh.NW[object][o]
		}
	}
	// Compute error.
	for o = 0; o < sh.no; o++ {
		// Discrepancy between object and its reconstruction.
		for f = 0; f < sh.nf; f++ {
			E += math.Pow(sh.O[o][f]-(sh.WP[o][f]/sh.W[o]), 2.)
		}
	}
	E /= float64(sh.no)
	return
}