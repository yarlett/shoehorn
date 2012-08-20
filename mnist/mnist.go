package main

import (
	"fmt"
	"shoehorn"
	"time"
)

func main() {
	// Load MNIST data.
	t1 := time.Now()
	sh := shoehorn.NewShoehorn("../data/mnist_data_train.csv", 2, 0.005)
	fmt.Printf("Took %v to create data set of %d objects.\n", time.Now().Sub(t1), len(sh.ObjectIDs()))

	// Perform gradient-descent with L2 punishment initially.
	sh.LearnL2(0.01, 0.01, 1000, 0.01, "tmp/locations_l2")

	// Perform radius-limited gradient descent initially.
	//sh.LearnRadius(0.5, 3.0, 1000, 0.01, "tmp/locations_radius")

	// // Perform Rprop learning to refine object locations.
	// sh.LearnRprop(0.01, 2000, 0.01, "tmp/locations_rprop")
}