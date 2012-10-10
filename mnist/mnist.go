package main

import (
	"fmt"
	"github.com/yarlett/shoehorn"
	"time"
)

func main() {
	// Load MNIST data.
	t1 := time.Now()
	sh := shoehorn.NewShoehorn("../data/mnist_data_train.csv", 2, 0.005)
	sh.NormalizeObjects(2.0)
	fmt.Printf("Took %v to create data set of %d objects exhibiting %d distinct features.\n", time.Now().Sub(t1), len(sh.O), len(sh.O[0]))

	// Perform simulated annealing learning.
	temp0, temp1, temp_decay, sigma, l2, output_prefix := 1e-2, 1e-7, 0.99, 0.01, 0.05, "tmp/locations"
	sh.LearnSimulatedAnnealingByObject(temp0, temp1, temp_decay, sigma, l2, output_prefix)
}