package shoehorn

import (
	"fmt"
	"math/rand"
	"time"
)

func (sh *Shoehorn) LearnRepositioning(epochs int, output_prefix string) {
	var (
		epoch  int
		l2, energy float64
		TM, tm time.Time
	)
	l2 = 0.
	TM = time.Now()
	// Perform learning.
	energy = sh.TotalEnergy(l2)
	for epoch = 0; epoch < epochs; epoch++ {
		tm = time.Now()
		energy = UpdaterGAT(energy, l2, sh)
		// Write locations to file.
		if output_prefix != "" {
			sh.WriteLocations(fmt.Sprintf("%v.csv", output_prefix))
		}
		// Report status.
		fmt.Printf("Epoch %d: Energy=%.6e (took %v; %v elapsed).\n", epoch, energy, time.Now().Sub(tm), time.Now().Sub(TM))
	}
}

func UpdaterGAT(current_energy, l2 float64, sh *Shoehorn) (energy float64) {
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
	// Get energy at new location.
	for d = 0; d < sh.nd; d++ {
		sh.L[o][d] = newloc[d]
	}
	new_energy = sh.TotalEnergy(l2)
	// Decide whether to keep new location.
	if new_energy < current_energy {
		energy = new_energy
	} else {
		for d = 0; d < sh.nd; d++ {
			sh.L[o][d] = curloc[d]
		}
		energy = current_energy
	}
	return
}