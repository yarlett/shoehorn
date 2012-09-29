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

	alpha := 0.0

	// Define parameters.
	sh.LearnGradientDescent(0.1, 0.01, alpha, 200, "tmp/locations1")
	sh.LearnLineSearch(0.01, alpha, 20, "tmp/locations2")
	sh.LearnGradientDescent(0.01, 0.0, alpha, 200, "tmp/locations3")
}