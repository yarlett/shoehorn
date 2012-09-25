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
	fmt.Printf("Took %v to create data set of %d objects.\n", time.Now().Sub(t1), len(sh.ObjectIDs()))

	// Define parameters.
	step_size, l2, alpha, numepochs := 0.01, 0.1, 0.0, 2500

	// Perform gradient-descent with L2 punishment initially.
	sh.Rescale(1e-6)
	sh.Learn(step_size, l2, alpha, numepochs, "tmp/locations")
}