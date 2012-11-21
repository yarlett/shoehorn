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

	// sh.Annealing(1e3, 1e-1, .9, 10, "tmp/locations")
	//sh.LearnRprop(1e-2, 0, 10000, "tmp/locations")
	sh.LearnGradientDescent(50., 0., 0., 10000, "tmp/locations")

	// sh.Foo(sh.O, sh.L, "tmp/locations2")
	// sh.LearnRepositioning(25000, "tmp/locations")
}