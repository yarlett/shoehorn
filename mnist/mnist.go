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

	alpha, l2 := 0.0, 0.05

	// Define parameters.
	sh.Rescale(1e-3)
	sh.LearnGradientDescent(1e-2, alpha, l2, 1000, "tmp/locations1")
	//sh.LearnLineSearch(alpha, l2, 30, "tmp/locations2")
}