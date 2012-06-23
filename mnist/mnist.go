package main

import (
	"fmt"
	"shoehorn"
	"time"
)

func main() {
	// Load MNIST data.
	t1 := time.Now()
	sh := shoehorn.NewShoehorn("../data/mnist_data_train_subset.csv", 2)
	fmt.Printf("Took %v to create data set of %d objects.\n", time.Now().Sub(t1), len(sh.ObjectIDs()))

	// Constant learning parameters.
	maxepochs, alpha, decay, error := 100, 0.001, "exp", "kl"

	// Perform first learning round.
	lr, momentum, exag, l2 := 0.1, 0.0, 100.0, 5.0
	sh.Learn(lr, momentum, maxepochs, alpha, exag, l2, decay, error)

	// // Perform second learning round.
	// lr, momentum, exag, l2 = 0.1, 0.8, 1.0, 0.0
	// sh.Learn(lr, momentum, maxepochs, alpha, exag, l2, decay, error)

	// Write learned locations to output file.
	sh.WriteLocations("mnist_locations.csv")
}