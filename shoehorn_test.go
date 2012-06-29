package shoehorn

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// Some general parameters.
var (
	NDIMS int     = 2
	KNN   int     = math.MaxInt32
	ALPHA float64 = 0.01
	L2    float64 = 0.0
)

// Returns a Shoehorn object initialized with some test data.
func GetTestData(nobjs, ndims int) (sh Shoehorn) {
	var (
		o, j                      int
		object_name, feature_name string
		feature_value             float64
	)
	sh = Shoehorn{ndims: ndims, feature_ixs: make(map[string]int), object_ixs: make(map[string]int)}
	// Generate random object data.
	for o = 0; o < nobjs; o++ {
		object_name = fmt.Sprintf("TestObject %v", o)
		for j = 0; j < ndims; j++ {
			feature_name = fmt.Sprintf("Feature %v", j)
			feature_value = rand.Float64()
			sh.Store(object_name, feature_name, feature_value)
		}
	}
	// Normalize sums of data.
	for o = 0; o < len(sh.objects); o++ {
		sum := sh.objects[o].Sum()
		for k, v := range sh.objects[o].data {
			sh.objects[o].Set(k, v/sum)
		}
	}
	return
}

// Checks that the gradient function and its approximation are close to one another.
func TestGradient(t *testing.T) {
	var (
		o, object_ix, j                   int
		E0, E1, nudge, approx_grad, pcerr float64
		G0                                [][]float64
		sh                                Shoehorn
	)
	nudge = 1e-10
	sh = GetTestData(50, 3)
	// Compute gradient information.
	sh.Gradients(KNN, ALPHA, L2)
	// Save error and gradient information.
	E0 = 0.0
	G0 = sh.GetObjectStore()
	for o = 0; o < len(sh.objects); o++ {
		E0 += sh.E[o]
		for j = 0; j < sh.ndims; j++ {
			G0[o][j] = sh.G[o][j]
		}
	}
	// Iterate over the position of each object in each dimension.
	for object_ix = 0; object_ix < len(sh.objects); object_ix++ {
		for j = 0; j < sh.ndims; j++ {
			// Nudge the position.
			sh.L[object_ix][j] += nudge
			// Calculate the new error.
			sh.Gradients(KNN, ALPHA, L2)
			E1 = 0.0
			for o = 0; o < len(sh.objects); o++ {
				E1 += sh.E[o]
			}
			// Approximate the gradient.
			approx_grad = (E1 - E0) / nudge
			// Reset the position.
			sh.L[object_ix][j] -= nudge
			// Compare actual and approximated gradients.
			pcerr = math.Abs((G0[object_ix][j]-approx_grad)/G0[object_ix][j]) * 100.0
			if pcerr > 0.01 {
				t.Errorf("Discrepancy in gradient for object %3d in dimension %3d: %3.2f%% Error: Analytic=%2.10e; Approximated=%2.10e.\n", object_ix, j, pcerr, G0[object_ix][j], approx_grad)
			}
		}
	}
}
