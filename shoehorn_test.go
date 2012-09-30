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
	ALPHA      float64 = 0.01
	L2         float64 = 1.0
)

// Returns a Shoehorn object initialized with some test data.
func GetTestData(nobjs, nd int) (sh Shoehorn) {
	var (
		o, j                      int
		S                         [][]string
	)
	// Generate random object data.
	S = make([][]string, 0)
	for o = 0; o < nobjs; o++ {
		s := make([]string, 3)
		s[0] = fmt.Sprintf("TestObject %v", o)
		for j = 0; j < nd; j++ {
			s[1] = fmt.Sprintf("Feature %v", j)
			s[2] = fmt.Sprintf("%v", rand.Float64())
			S = append(S, s)
		}
	}
	// Create showhoen instance from the data.
	sh = Shoehorn{}
	sh.Create(S, nd)
	sh.NormalizeObjects(1.0)
	return
}

// Test that the object reconstructions are normalized discrete probability distributions.
func TestReconstructions(t *testing.T) {
	// Initialize the test data.
	sh := GetTestData(50, 3)
	// Get the object reconstruction data.
	sh.SetReconstructions()
	// Check the reconstruction of each object.
	for o := 0; o < sh.no; o++ {
		sump, sumq := 0.0, 0.0
		for f, p := range sh.O[o] {
			q := (ALPHA * p) + ((1.0 - ALPHA) * (sh.WP[o][f] / sh.W[o]))
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
		o, j                        int
		approx_grad, pcerr, h, Eh, Enh float64
		G                         [][]float64
		sh                             Shoehorn
	)
	// Initialize test data.
	sh = GetTestData(50, 3)
	h = 1e-6 // Step size used when approximating gradient.
	// Compute and save gradient information for objects.
	sh.SetGradients(ALPHA, L2)
	G = sh.CopyGradient()
	// Iterate over the position of each object in each dimension.
	for o = 0; o < sh.no; o++ {
		for j = 0; j < sh.nd; j++ {
			// Calculate error at x - h.
			sh.L[o][j] -= h
			sh.SetErrors(ALPHA, L2)
			Enh = sh.CurrentError() * float64(sh.no)
			// Calculate error at x + h.
			sh.L[o][j] += 2.0 * h
			sh.SetErrors(ALPHA, L2)
			Eh = sh.CurrentError() * float64(sh.no)
			// Reset x to original position.
			sh.L[o][j] -= h
			// Calculate approximate gradient.
			approx_grad = (Eh - Enh) / (2.0 * h)
			// Compare actual and approximated gradients.
			pcerr = math.Abs((G[o][j]-approx_grad)/G[o][j]) * 100.0
			if pcerr > 0.05 {
				t.Errorf("Discrepancy in gradient for object %3d in dimension %3d: %3.6f%% Error: h=%e: Analytic=%2.10e; Approximated=%2.10e.\n", o, j, pcerr, h, G[o][j], approx_grad)
			}
		}
	}
}
