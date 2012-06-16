package shoehorn

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"
)

type Shoehorn struct {
	ndims       int
	locs        [][]float64
	object_ixs  map[string]int
	feature_ixs map[string]int
	ftctr       int
	objects     []*FeatureVector
}

//
// Store method. Stores feature value for an object, and updates internal indices.
//

func (sh *Shoehorn) Store(object string, feature string, value float64) {
	// Update object indices.
	_, obfound := sh.object_ixs[object]
	if !obfound {
		// Create new feature vector and location for the object if it hasn't been encountered before.
		sh.objects = append(sh.objects, &FeatureVector{data: make(map[int]float64)})
		loc := make([]float64, sh.ndims)
		for i := 0; i < sh.ndims; i++ {
			loc[i] = rand.NormFloat64() * 0.001
		}
		sh.locs = append(sh.locs, loc)
		sh.object_ixs[object] = len(sh.objects) - 1
	}
	// Update feature indices.
	_, ftfound := sh.feature_ixs[feature]
	if !ftfound {
		sh.feature_ixs[feature] = sh.ftctr
		sh.ftctr++
	}
	// Store the feature value for the object.
	object_ix, _ := sh.object_ixs[object]
	feature_ix, _ := sh.feature_ixs[feature]
	sh.objects[object_ix].Set(feature_ix, value)
}

//
// Learning method.
//

func (sh *Shoehorn) Learn(knn int, alpha float64, lr float64, maxepochs int) {
	var (
		epoch, object_ix, chanctr, j                                int
		mE0, mE, mG, magG, l2_coeff, exag, uselr, updatej, momentum float64
		T, t                                                        time.Time
		gradient_channel                                            chan GradientInfo
		G                                                           []GradientInfo
		U                                                           [][]float64
	)
	// Initialize storage for last update.
	U = make([][]float64, len(sh.objects))
	for object_ix = 0; object_ix < len(sh.objects); object_ix++ {
		U[object_ix] = make([]float64, sh.ndims)
	}
	// Set the maximum number of threads to be used to the number of CPU cores available.
	numprocs := runtime.NumCPU()
	runtime.GOMAXPROCS(numprocs)
	// Initialize learning parameters.
	exag = 1.0
	l2_coeff = 0.00
	momentum = 0.25
	uselr = lr
	// Iterate over epochs of learning.
	T = time.Now()
	for epoch, mG = 0, math.MaxFloat64; epoch < maxepochs; epoch++ {
		t = time.Now()
		// Switch to later learning parameters if appropriate.
		if epoch == maxepochs/1 {
			exag = 1.0
			l2_coeff = 0.0
			momentum = 0.8
			uselr = lr
			for object_ix = 0; object_ix < len(sh.objects); object_ix++ {
				for j = 0; j < sh.ndims; j++ {
					U[object_ix][j] = 0.0
				}
			}
		}
		// Create a channel for gradient information to be returned on.
		gradient_channel = make(chan GradientInfo, len(sh.objects))
		// Calculate gradient information for all objects using goroutines.
		for object_ix = 0; object_ix < len(sh.objects); object_ix++ {
			go sh.GradientPow(object_ix, sh.locs[object_ix], knn, alpha, uselr, l2_coeff, exag, gradient_channel)
		}
		// Collect all the gradient information from the gradient channel.
		G = make([]GradientInfo, 0)
		for chanctr = 0; chanctr < len(sh.objects); chanctr++ {
			G = append(G, <-gradient_channel)
		}
		close(gradient_channel)
		// Update positions based on gradient information.
		mE = 0.0
		mG = 0.0
		for chanctr = 0; chanctr < len(sh.objects); chanctr++ {
			// Update error.
			mE += G[chanctr].error
			// Update overall magnitude of gradient.
			magG = 0.0
			for j = 0; j < sh.ndims; j++ {
				magG += math.Pow(G[chanctr].gradient[j], 2.0)
			}
			mG += math.Pow(magG, 0.5)
			// Perform gradient descent using momentum.
			for j = 0; j < sh.ndims; j++ {
				updatej = (uselr * G[chanctr].gradient[j]) + (momentum * U[G[chanctr].object_ix][j])
				sh.locs[G[chanctr].object_ix][j] -= updatej
				U[G[chanctr].object_ix][j] = updatej
			}
		}
		mE /= float64(len(sh.objects))
		mG /= float64(len(sh.objects))
		// Report performance for epoch.
		fmt.Printf("Epoch %6d: mE=%.10e mG=%.10e (lr=%e, mom=%e, odist=%e, epoch took %v; elapsed %v).\n", epoch, mE, mG, uselr, momentum, sh.OriginDistance(), time.Now().Sub(t), time.Now().Sub(T))
		// Update learning rate using "bold driver" method.
		if epoch > 0 {
			if mE < mE0 {
				// Error decreased, so increase learning rate a little.
				uselr *= 1.05
			} else {
				// Error increased, so undo the step that increased the error; reset the momentum term; and reduce the learning rate.
				for chanctr = 0; chanctr < len(G); chanctr++ {
					for j = 0; j < sh.ndims; j++ {
						sh.locs[G[chanctr].object_ix][j] += U[G[chanctr].object_ix][j]
						U[G[chanctr].object_ix][j] = 0.0
					}
				}
				uselr *= 0.5
			}
		}
		// Update the previous error.
		mE0 = mE
	}
}

// //
// // Error method.
// //

// func (sh *Shoehorn) Error(object_ix int, object_loc []float64, knn int, alpha float64, l2_coeff float64, exag float64) (E float64) {
// 	var (
// 		j       int
// 		q, Q, W float64
// 		N       Weights
// 	)
// 	// Get nearest neighbors and sum of weights.
// 	N, W = sh.NeighborsExp(object_ix, object_loc, knn)
// 	// Iterate over features in the object.
// 	for feature_ix, p := range sh.objects[object_ix].data {
// 		p *= exag
// 		// Compute weighted sum over the nearest neighbors.
// 		Q = 0.0
// 		for _, n := range N {
// 			Q += n.weight * sh.objects[n.object_ix].data[feature_ix]
// 		}
// 		// Finalize the reconstructed probability estimate.
// 		q = (alpha * p) + ((1.0 - alpha) * (Q / W))
// 		// Update the error.
// 		E += (p * (math.Log(p) - math.Log(q)))
// 	}
// 	// Add distance from origin penalty.
// 	for j = 0; j < sh.ndims; j++ {
// 		E += (l2_coeff * object_loc[j] * object_loc[j])
// 	}
// 	return
// }

//
// Gradient method.
//

func (sh *Shoehorn) GradientExp(object_ix int, object_loc []float64, knn int, alpha float64, lr float64, l2_coeff float64, exag float64, gradient_channel chan GradientInfo) {
	var (
		j, feature_ix             int
		W, E, p, q, Q, tmp1, tmp2 float64
		G, T1, T2                 []float64
		N                         Weights
		n                         WeightPair
	)
	// Perform initializations.
	G = make([]float64, sh.ndims)
	T1 = make([]float64, sh.ndims)
	T2 = make([]float64, sh.ndims)
	// Get nearest neighbors and sum of weights.
	N, W = sh.NeighborsExp(object_ix, object_loc, knn)
	// Iterate over features of object.
	for feature_ix, p = range sh.objects[object_ix].data {
		// Account for exaggeration factor.
		p *= exag
		// Reset values of accumulating terms.
		Q = 0.0
		for j = 0; j < sh.ndims; j++ {
			T1[j], T2[j] = 0.0, 0.0
		}
		// Iterate over nearest neighbors and update statistics.
		for _, n = range N {
			Q += n.weight * sh.objects[n.object_ix].data[feature_ix]
			tmp1 = n.weight / n.distance
			for j = 0; j < sh.ndims; j++ {
				tmp2 = (sh.locs[n.object_ix][j] - object_loc[j]) * tmp1
				T1[j] += tmp2 * sh.objects[n.object_ix].data[feature_ix]
				T2[j] += tmp2
			}
		}
		// Set the reconstruction probability.
		q = (alpha * p) + ((1.0 - alpha) * (Q / W))
		// Update gradient information.
		for j = 0; j < sh.ndims; j++ {
			G[j] += ((alpha - 1.0) * p / q) * (((T1[j] * W) - (Q * T2[j])) / (W * W))
		}
		// Update the error.
		E += (p * (math.Log(p) - math.Log(q)))
	}

	// // Add distance penalty to gradient.
	// ssd := 0.0
	// for j = 0; j < sh.ndims; j++ {
	// 	ssd += math.Pow(object_loc[j], 2.0)
	// }
	// E += l2_coeff * math.Pow(ssd, 0.5)
	// for j = 0; j < sh.ndims; j++ {
	// 	G[j] += l2_coeff * object_loc[j] * math.Pow(ssd, -0.5)
	// }

	// // Add distance penalty to gradient.
	// for j = 0; j < sh.ndims; j++ {
	// 	E += (l2_coeff * object_loc[j] * object_loc[j])
	// 	G[j] += (2.0 * l2_coeff * object_loc[j])
	// }

	// Return gradient information.
	gradient_channel <- GradientInfo{object_ix: object_ix, gradient: G, error: E, lr: lr}
}

func (sh *Shoehorn) GradientPow(object_ix int, object_loc []float64, knn int, alpha float64, lr float64, l2_coeff float64, exag float64, gradient_channel chan GradientInfo) {
	var (
		j, feature_ix int
		W, E, p, q, Q, tmp float64
		G, T1, T2     []float64
		N             Weights
		n             WeightPair
	)
	// Perform initializations.
	G = make([]float64, sh.ndims)
	T1 = make([]float64, sh.ndims)
	T2 = make([]float64, sh.ndims)
	// Get nearest neighbors and sum of weights.
	N, W = sh.NeighborsPow(object_ix, object_loc, knn)
	// Iterate over features of object.
	for feature_ix, p = range sh.objects[object_ix].data {
		// Account for exaggeration factor.
		p *= exag
		// Reset values of accumulating terms.
		Q = 0.0
		for j = 0; j < sh.ndims; j++ {
			T1[j], T2[j] = 0.0, 0.0
		}
		// Iterate over nearest neighbors and update statistics.
		for _, n = range N {
			Q += n.weight * sh.objects[n.object_ix].data[feature_ix]
			for j = 0; j < sh.ndims; j++ {
				tmp = (sh.locs[n.object_ix][j] - object_loc[j]) * math.Pow(math.Pow(n.d, 0.5) + 1.0, -2.0) * math.Pow(n.d, -0.5)
				T1[j] += tmp * sh.objects[n.object_ix].data[feature_ix]
				T2[j] += tmp
			}
		}
		// Set the reconstruction probability.
		q = (alpha * p) + ((1.0 - alpha) * (Q / W))
		// Update gradient information.
		for j = 0; j < sh.ndims; j++ {
			G[j] += ((alpha - 1.0) * p / q) * (((T1[j] * W) - (Q * T2[j])) / (W * W))
		}
		// Update the error.
		E += (p * (math.Log(p) - math.Log(q)))
	}

	// // Add distance penalty to gradient.
	// ssd := 0.0
	// for j = 0; j < sh.ndims; j++ {
	// 	ssd += math.Pow(object_loc[j], 2.0)
	// }
	// E += l2_coeff * math.Pow(ssd, 0.5)
	// for j = 0; j < sh.ndims; j++ {
	// 	G[j] += l2_coeff * object_loc[j] * math.Pow(ssd, -0.5)
	// }

	// // Add distance penalty to gradient.
	// for j = 0; j < sh.ndims; j++ {
	// 	E += (l2_coeff * object_loc[j] * object_loc[j])
	// 	G[j] += (2.0 * l2_coeff * object_loc[j])
	// }

	// Return gradient information.
	gradient_channel <- GradientInfo{object_ix: object_ix, gradient: G, error: E, lr: lr}
}

//
// Utility methods.
//

func (sh *Shoehorn) ObjectIDs() (object_ids []int) {
	for id := 0; id < len(sh.objects); id++ {
		object_ids = append(object_ids, id)
	}
	return
}

func (sh *Shoehorn) NeighborsExp(object_ix int, object_loc []float64, knn int) (N Weights, W float64) {
	N = sh.WeightsExp(object_ix, object_loc)
	if (knn > 0) && (knn < len(N)) {
		N = N[:knn]
	}
	for _, n := range N {
		W += n.weight
	}
	return
}

func (sh *Shoehorn) NeighborsPow(object_ix int, object_loc []float64, knn int) (N Weights, W float64) {
	N = sh.WeightsPow(object_ix, object_loc)
	if (knn > 0) && (knn < len(N)) {
		N = N[:knn]
	}
	for _, n := range N {
		W += n.weight
	}
	return
}

func (sh *Shoehorn) WeightsExp(object_ix int, object_loc []float64) (weights Weights) {
	var (
		w *WeightPair
		o int
	)
	for o = 0; o < len(sh.objects); o++ {
		w = new(WeightPair)
		// Set the object.
		w.object_ix = o
		// Set distance.
		w.d = 0.0
		for j := 0; j < sh.ndims; j++ {
			w.d += math.Pow(object_loc[j]-sh.locs[o][j], 2.0)
		}
		w.distance = math.Pow(w.d, 0.5)
		// Set weight.
		w.weight = math.Exp(-w.distance)
		// Store weight information (don't count a point as its own neighbor).
		if w.distance > 0.0 {
			weights = append(weights, *w)
		}
	}
	sort.Sort(weights)
	return
}

func (sh *Shoehorn) WeightsPow(object_ix int, object_loc []float64) (weights Weights) {
	var (
		w *WeightPair
		o int
	)
	for o = 0; o < len(sh.objects); o++ {
		w = new(WeightPair)
		// Set the object.
		w.object_ix = o
		// Set distance.
		w.d = 0.0
		for j := 0; j < sh.ndims; j++ {
			w.d += math.Pow(object_loc[j]-sh.locs[o][j], 2.0)
		}
		w.distance = math.Pow(w.d, 0.5)
		// Set weight.
		w.weight = math.Pow(1.0 + w.distance, -1.0)
		// Store weight information (don't count a point as its own neighbor).
		if w.distance > 0.0 {
			weights = append(weights, *w)
		}
	}
	sort.Sort(weights)
	return
}

// Writes the current locations of objects to a file.
func (sh *Shoehorn) WriteLocations(path string) {
	// Initialize the output file.
	of, err := os.Create(path)
	if err != nil {
		log.Fatal(err)
	}
	defer of.Close()
	// Write object locations to file.
	for object_name, object_ix := range sh.object_ixs {
		line := make([]string, 0)
		line = append(line, object_name)
		for i := 0; i < sh.ndims; i++ {
			line = append(line, fmt.Sprintf("%v", sh.locs[object_ix][i]))
		}
		of.Write([]byte(fmt.Sprintf("%v\n", strings.Join(line, ","))))
	}
}

// Returns the average distance of objects from the origin.
func (sh *Shoehorn) OriginDistance() (dist float64) {
	var n float64
	for o := 0; o < len(sh.objects); o++ {
		d := 0.0
		for j := 0; j < sh.ndims; j++ {
			d += sh.locs[o][j] * sh.locs[o][j]
		}
		d = math.Pow(d, 0.5)
		dist += d
		n += 1.0
	}
	dist = dist / n
	return
}

//
// Constructs a DataSet from a CSV file of {object_name, feature_name, feature_value} triples.
//

func NewShoehorn(filename string, ndims int) (sh *Shoehorn) {
	var (
		bfr *bufio.Reader
	)
	// Initialize the data set.
	sh = &Shoehorn{ndims: ndims, feature_ixs: make(map[string]int), object_ixs: make(map[string]int)}
	// Open the file for reading.
	fh, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer fh.Close()
	// Read the lines of the file one at a time.
	bfr = bufio.NewReaderSize(fh, 1024*16)
	for line, isprefix, err := bfr.ReadLine(); err != io.EOF; {
		// Error handling.
		if err != nil {
			log.Fatal(err)
		}
		if isprefix {
			log.Fatal("Line too long for buffered reader.")
		}
		// Extract the three values on the line.
		strvals := strings.Split(string(line), ",")
		if len(strvals) == 3 {
			value_float, _ := strconv.ParseFloat(strvals[2], 64)
			// Store the data in the data set.
			sh.Store(strvals[0], strvals[1], value_float)
		}
		// Read from the file for the next iteration.
		line, isprefix, err = bfr.ReadLine()
	}
	// Normalize each vector so they sum to 1.
	for object_ix := 0; object_ix < len(sh.objects); object_ix++ {
		sum := sh.objects[object_ix].Sum()
		for k, v := range sh.objects[object_ix].data {
			sh.objects[object_ix].Set(k, v/sum)
		}
	}
	return
}
