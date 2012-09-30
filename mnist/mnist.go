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

	alpha, l2 := 0.0, 0.0

	// Define parameters.
	sh.LearnGradientDescent(0.1, alpha, l2, 300, "tmp/locations1")
	sh.Rescale(1.0)
	sh.LearnLineSearch(alpha, l2, 30, "tmp/locations2")
}