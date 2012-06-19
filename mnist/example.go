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
	// Set the learning parameters.
	knn, alpha, lr, maxepochs, decay := 1000, 0.0001, 0.1, 250, "exp"
	// Learn locations based on parameters.
	sh.Learn(knn, alpha, lr, maxepochs, decay)
	// Write learned locations to output file.
	sh.WriteLocations("mnist_locations.csv")
}