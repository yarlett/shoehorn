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
	fmt.Printf("Took %v to create data set of %d objects exhibiting %d distinct features.\n", time.Now().Sub(t1), len(sh.O), len(sh.O[0]))

	//sh.LearnRprop(.01, 0., 10000, "tmp/locations")

	sh.LearnGradientDescent(1e-7, 0, 0, 10000, "tmp/locations")
	// sh.LearnGradientDescent(25., 0.9, 0., 5000, "tmp/locations")
	// sh.LearnRepositioning(250000, "tmp/locations")
}