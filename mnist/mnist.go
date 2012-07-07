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
	alpha := 0.01

	// Assign initial positions.
	sh.InitialPositions(5, alpha)
	sh.WriteLocations("mnist_locations1.csv")

	// // Perform first round of gradient descent.
	// lr, momentum, l2, numepochs := 0.1, 0.75, 0.0, 100
	// sh.Learn(lr, momentum, l2, numepochs, alpha)
	// sh.WriteLocations("mnist_locations1.csv")

	// // Perform repositioning search.
	// cycles, knn := 3, 15
	// sh.RepositioningSearch(cycles, knn, alpha)
	// sh.WriteLocations("mnist_locations2.csv")

	// // Perform second round of gradient descent.
	// sh.Learn(lr, momentum, numepochs, alpha)
	// sh.WriteLocations("mnist_locations3.csv")
}