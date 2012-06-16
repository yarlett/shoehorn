package main

import (
	"fmt"
	"shoehorn"
	"time"
)

func main() {
	// Load data.
	t1 := time.Now()
	sh := shoehorn.NewShoehorn("../data/mnist_data_train_subset.csv", 2)
	fmt.Printf("Took %v to create data set of %d objects.\n", time.Now().Sub(t1), len(sh.ObjectIDs()))

	// Learning parameters.
	knn, alpha, lr, maxepochs := 1000, 0.01, 0.01, 100

	// Learn locations based on parameters.
	sh.Learn(knn, alpha, lr, maxepochs)

	// Write learned locations to output file.
	sh.WriteLocations("mnist_locations.csv")
}