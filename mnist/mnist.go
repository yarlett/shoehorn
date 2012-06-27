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
	alpha := 0.01

	// Perform first round of gradient descent.
	lr, momentum, numepochs := 5.0, 0.8, 50
	sh.Learn(lr, momentum, numepochs, alpha)
	sh.WriteLocations("mnist_locations1.csv")

	// // Perform repositioning search.
	// cycles, knn := 3, 15
	// sh.RepositioningSearch(cycles, knn, alpha)
	// sh.WriteLocations("mnist_locations2.csv")

	// // Perform second round of gradient descent.
	// sh.Learn(lr, momentum, numepochs, alpha)
	// sh.WriteLocations("mnist_locations3.csv")
}