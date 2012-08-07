package main

import (
	"fmt"
	"shoehorn"
	"time"
)

func main() {
	// Load MNIST data.
	t1 := time.Now()
	//sh := shoehorn.NewShoehorn("../data/synthetic_data.csv", 2, 1.0)
	sh := shoehorn.NewShoehorn("../data/mnist_data_train.csv", 2, 0.005)
	fmt.Printf("Took %v to create data set of %d objects.\n", time.Now().Sub(t1), len(sh.ObjectIDs()))

	// Constant learning parameters.
	alpha := 0.01

	// Perform first round of gradient descent.
	max_move, momentum, l2, numepochs := 0.01, 0.0, 0.0, 2500
	sh.Learn(max_move, momentum, l2, numepochs, alpha, "tmp/locations")
}