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
	sh.Annealing(1e-1, 1e-6, 10, 1000, 0.1, 2e-2, "tmp/locations")
}