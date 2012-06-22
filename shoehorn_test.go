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
var EXAG float64 = 0.0

// Load some of the MNIST data.
var path, _ = os.Getwd()
var sh = NewShoehorn(strings.Join([]string{path, "/data/mnist_data_train_subset.csv"}, ""), NDIMS)

// Checks that the gradient function and its approximation are close to one another.
func TestGradient(t *testing.T) {
	var delta float64 = 1e-8
	var tol float64 = 1e-6
	for _, decay := range []string{"exp", "pow"} {
		for _, l2 := range []float64{0.0, 1.0} {
			for _, alpha := range []float64{0.5, 0.01} {
				for _, exag := range []float64{1.0, 10.0} {
					// Iterate over objects and accumulate error between analytic and approximated gradient.
					for object_ix := 0; object_ix < 10; object_ix++ {
						// Get the error and gradient from the gradient function.
						gradient_channel := make(chan GradientInfo, 1)
						sh.Gradient(object_ix, sh.locs[object_ix], KNN, alpha, l2, exag, decay, gradient_channel)
						g := <-gradient_channel
						E0 := g.error
						G0 := g.gradient
						// Approximate the gradient.
						G1 := make([]float64, NDIMS)
						for j := 0; j < NDIMS; j++ {
							sh.locs[object_ix][j] += delta
							gradient_channel := make(chan GradientInfo, 1)
							sh.Gradient(object_ix, sh.locs[object_ix], KNN, alpha, l2, exag, decay, gradient_channel)
							g := <-gradient_channel
							E1 := g.error
							G1[j] = (E1 - E0) / delta
							sh.locs[object_ix][j] -= delta
						}
						// Measure discrepenacy between gradients.
						for j := 0; j < NDIMS; j++ {
							if math.Abs(G0[j]-G1[j]) > tol {
								t.Errorf("Gradient error: decay=%s, l2=%e, alpha=%e, exag=%e: object=%v: G0[%v]=%v G1[%v]=%v", decay, l2, alpha, exag, object_ix, j, G0[j], j, G1[j])
							}
						}
					}
				}
			}
		}
	}
}