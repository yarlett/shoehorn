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

	// Define parameters.
	step_size, l2, alpha, numepochs := 0.01, 0.05, 0.0, 2500

	// Perform gradient-descent with L2 punishment initially.
	sh.Rescale(1e-6)
	sh.LearnGradientDescent(step_size, l2, alpha, numepochs, "tmp/locations")
}