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
		approx_grad, pcerr, h, Eh, Enh float64
		G0                                [][]float64
		sh                                Shoehorn
	)
	// Initialize test data.
	sh = GetTestData(50, 3)
	// Compute and save gradient information for objects.
	sh.Gradients(KNN, ALPHA, L2)
	G0 = sh.GetObjectStore()
	for o = 0; o < len(sh.objects); o++ {
		for j = 0; j < sh.ndims; j++ {
			G0[o][j] = sh.G[o][j]
		}
	}
	// Iterate over the position of each object in each dimension.
	for object_ix = 0; object_ix < len(sh.objects); object_ix++ {
		for j = 0; j < sh.ndims; j++ {
			// Calculate the step size h (small enoguh to provide good approximation to gradient without being so small as to cause underflow / rounding issues in calculation).
			h = math.Pow(2.2e-16, 0.5) * sh.L[object_ix][j]
			// Calculate error at x - h.
			sh.L[object_ix][j] -= h
			sh.Gradients(KNN, ALPHA, L2)
			Enh = 0.0
			for o = 0; o < len(sh.objects); o++ {
				Enh += sh.E[o]
			}
			// Calculate error at x + h.
			sh.L[object_ix][j] += 2.0 * h
			sh.Gradients(KNN, ALPHA, L2)
			Eh = 0.0
			for o = 0; o < len(sh.objects); o++ {
				Eh += sh.E[o]
			}
			// Reset x to original position.
			sh.L[object_ix][j] -= h
			// Calculate approximate gradient.
			approx_grad = (Eh - Enh) / (2.0 * h)
			// Compare actual and approximated gradients.
			pcerr = math.Abs((G0[object_ix][j]-approx_grad)/G0[object_ix][j]) * 100.0
			if pcerr > 0.1 {
				t.Errorf("Discrepancy in gradient for object %3d in dimension %3d: %3.2f%% Error: h=%e: Analytic=%2.10e; Approximated=%2.10e.\n", object_ix, j, pcerr, h, G0[object_ix][j], approx_grad)
			}
		}
	}
}
