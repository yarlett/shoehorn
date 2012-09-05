package shoehorn

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// Some general parameters.
var (
	NDIMS      int     = 2
	MIN_WEIGHT float64 = 0.0
	ALPHA      float64 = 0.01
	L2         float64 = 1.0
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
	sh.NormalizeObjects(1.0)
	return
}

// Test that the object reconstructions are normalized discrete probability distributions.
func TestReconstructions(t *testing.T) {
	// Initialize the test data.
	sh := GetTestData(50, 3)
	// Get the object reconstruction data.
	R := sh.Reconstructions(MIN_WEIGHT)
	// Check the reconstruction of each object.
	for o := 0; o < sh.nobjs; o++ {
		sump, sumq := 0.0, 0.0
		for f, p := range sh.objects[o].data {
			q := (ALPHA * p) + ((1.0 - ALPHA) * (R.WPS[o][f] / R.WS[o]))
			sump += p
			sumq += q
		}
		if (math.Abs(sump-1.0) > 1e-10) || (math.Abs(sumq-1.0) > 1e-10) {
			t.Errorf("Problem with reconstruction of object %d: sump=%e sumq=%e.\n", o, sump, sumq)
		}
	}
}

// Checks that the gradient function and its approximation are close to one another.
func TestGradient(t *testing.T) {
	var (
		o, j, k                        int
		approx_grad, pcerr, h, Eh, Enh float64
		G0, G1                         []GradientInfo
		sh                             Shoehorn
	)
	// Initialize test data.
	sh = GetTestData(50, 3)
	h = 1e-6 // Step size used when approximating gradient.
	// Compute and save gradient information for objects.
	G0 = sh.Gradients(MIN_WEIGHT, ALPHA, L2)
	// Iterate over the position of each object in each dimension.
	for o = 0; o < sh.nobjs; o++ {
		for j = 0; j < sh.ndims; j++ {
			// Calculate error at x - h.
			sh.L[o][j] -= h
			G1 = sh.Gradients(MIN_WEIGHT, ALPHA, L2)
			Enh = 0.0
			for k = 0; k < len(G1); k++ {
				Enh += G1[k].error
			}
			//Enh = sh.Error(MIN_WEIGHT, ALPHA, L2)
			// Calculate error at x + h.
			sh.L[o][j] += 2.0 * h
			G1 = sh.Gradients(MIN_WEIGHT, ALPHA, L2)
			Eh = 0.0
			for k = 0; k < len(G1); k++ {
				Eh += G1[k].error
			}
			//Eh = sh.Error(MIN_WEIGHT, ALPHA, L2)
			// Reset x to original position.
			sh.L[o][j] -= h
			// Calculate approximate gradient.
			approx_grad = (Eh - Enh) / (2.0 * h)
			// Compare actual and approximated gradients.
			pcerr = math.Abs((G0[o].gradient[j]-approx_grad)/G0[o].gradient[j]) * 100.0
			if pcerr > 0.05 {
				t.Errorf("Discrepancy in gradient for object %3d in dimension %3d: %3.6f%% Error: h=%e: Analytic=%2.10e; Approximated=%2.10e.\n", o, j, pcerr, h, G0[o].gradient[j], approx_grad)
			}
		}
	}
}
