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

	// Perform radius-limited gradient descent initially.
	sh.LearnRadius(0.1, 10.0, 2000, 0.01, "tmp/locations_radius")

	// Perform Rprop learning to refine object locations.
	sh.LearnRprop(0.01, 2000, 0.01, "tmp/locations_rprop")
}