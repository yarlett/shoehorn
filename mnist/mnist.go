package main

import (
	"fmt"
	"github.com/yarlett/shoehorn"
	"time"
)

func main() {
	// Load MNIST data.
	t1 := time.Now()
	sh := shoehorn.NewShoehorn("../data/mnist_data_train.csv", 2, 0.01)
	fmt.Printf("Took %v to create data set of %d objects exhibiting %d distinct features.\n", time.Now().Sub(t1), len(sh.O), len(sh.O[0]))

	sh.TransformData()
	sh.LearnGradientDescent(1e-3, 0.0, 2e1, .25, 5000, "mnist")
	//sh.LearnGradientDescent(1e-7, 0.5, 0e0, .25, 3000, "mnist2")
}