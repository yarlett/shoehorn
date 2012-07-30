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

	// Constant learning parameters.
	alpha := 0.001

	// Perform first round of gradient descent.
	max_move, momentum, l2, numepochs := 0.1, 0.0, 0.0, 1000
	sh.LearnSingleUpdate(max_move, momentum, l2, numepochs, alpha, "tmp/mnist_locations")
}