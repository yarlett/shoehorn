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
	//alpha := 0.01

	// Perform first round of gradient descent.
	//max_move, momentum, l2 := 0.005, 0.0, 0.0
	momentum, alpha, l2 := 0.0, 0.01, 0.1
	sh.Learn(0.01, momentum, 100000, alpha, l2, "tmp/locations")
	//sh.Learn(0.1, momentum, 500, alpha, false, "tmp/locations")
}