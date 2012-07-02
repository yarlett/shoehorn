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
	sh.NormalizeObjectSums()
	return
}

// Checks that the gradient function and its approximation are close to one another.
func TestGradient(t *testing.T) {
	var (
		o, j                           int
		approx_grad, pcerr, h, Eh, Enh float64
		G                              [][]float64
		sh                             Shoehorn
	)
	// Initialize test data.
	sh = GetTestData(50, 3)
	h = 1e-6 // Step size used when approximating gradient.
	// Compute and save gradient information for objects.
	for o = 0; o < len(sh.objects); o++ {
		G = append(G, sh.Gradient(o, KNN, ALPHA, L2))
	}
	// Iterate over the position of each object in each dimension.
	for o = 0; o < len(sh.objects); o++ {
		for j = 0; j < sh.ndims; j++ {
			// Calculate error at x - h.
			sh.L[o][j] -= h
			Enh = sh.Error(KNN, ALPHA, L2)
			// Calculate error at x + h.
			sh.L[o][j] += 2.0 * h
			Eh = sh.Error(KNN, ALPHA, L2)
			// Reset x to original position.
			sh.L[o][j] -= h
			// Calculate approximate gradient.
			approx_grad = (Eh - Enh) / (2.0 * h)
			// Compare actual and approximated gradients.
			pcerr = math.Abs((G[o][j]-approx_grad)/G[o][j]) * 100.0
			if pcerr > 0.1 {
				t.Errorf("Discrepancy in gradient for object %3d in dimension %3d: %3.2f%% Error: h=%e: Analytic=%2.10e; Approximated=%2.10e.\n", o, j, pcerr, h, G[o][j], approx_grad)
			}
		}
	}
}
