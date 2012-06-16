package shoehorn

import (
	"math"
	"os"
	"strings"
	"testing"
)

// Some general parameters.
var NDIMS int = 2
var KNN int = math.MaxInt32
var ALPHA float64 = 0.01

// Load some of the MNIST data.
var path, _ = os.Getwd()
var sh = NewShoehorn(strings.Join([]string{path, "/data/mnist_data_train_subset.csv"}, ""), NDIMS)

func TestErrorFuncs(t *testing.T) {
	for object_ix := 0; object_ix < len(sh.objects); object_ix++ {
		// Get the error from the error function.
		E1 := sh.Error(object_ix, KNN, ALPHA)
		// Get the error from the gradient function.
		gradient_channel := make(chan GradientInfo, 1)
		sh.Gradient(object_ix, KNN, ALPHA, gradient_channel)
		g := <-gradient_channel
		E2 := g.error
		// Report error if the values differ non-negligibly.
		if math.Abs(E1-E2) > 1e-12 {
			t.Errorf("Errors for object %d do not agree (%v, vs. %v).", object_ix, E1, E2)
		}
	}
}

// Checks that the gradient function and its approximation are close to one another.
func TestGradient(t *testing.T) {
	var delta float64 = 1e-8
	var tol float64 = 1e-6
	// Iterate over objects and accumulate error between analytic and approximated gradient.
	for object_ix := 0; object_ix < len(sh.objects); object_ix++ {
		// Get the gradient from the gradient function.
		gradient_channel := make(chan GradientInfo, 1)
		sh.Gradient(object_ix, KNN, ALPHA, gradient_channel)
		g := <-gradient_channel
		G1 := g.gradient
		// Approximate the gradient.
		E0 := sh.Error(object_ix, KNN, ALPHA)
		G2 := make([]float64, NDIMS)
		for i := 0; i < NDIMS; i++ {
			sh.locs[object_ix][i] += delta
			E1 := sh.Error(object_ix, KNN, ALPHA)
			G2[i] = (E1 - E0) / delta
			sh.locs[object_ix][i] -= delta
		}
		// Measure discrepenacy between gradients.
		for i := 0; i < NDIMS; i++ {
			if math.Abs(G1[i]-G2[i]) > tol {
				t.Errorf("Gradient error: object=%v: G1[%v]=%v G2[%v]=%v", object_ix, i, G1[i], i, G2[i])
			}
		}
	}
}
