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

func (sh *Shoehorn) Learn(alpha float64, lr float64, maxepochs int, decay string) {
	var (
		knn, epoch, object_ix, chanctr, j                  int
		mE0, mE, mG, magG, l2, exag, uselr, momentum float64
		T, t                                               time.Time
		gradient_channel                                   chan GradientInfo
		G                                                  []GradientInfo
		U0, U1                                             [][]float64
	)
	// Initialize storage for last and current update.
	U0 = make([][]float64, len(sh.objects))
	U1 = make([][]float64, len(sh.objects))
	for object_ix = 0; object_ix < len(sh.objects); object_ix++ {
		U0[object_ix] = make([]float64, sh.ndims)
		U1[object_ix] = make([]float64, sh.ndims)
	}
	// Set the maximum number of threads to be used to the number of CPU cores available.
	numprocs := runtime.NumCPU()
	runtime.GOMAXPROCS(numprocs)
	// Initialize learning parameters.
	knn = len(sh.objects)
	exag = 5.0
	l2 = 0.2
	momentum = 0.2
	uselr = lr
	// Iterate over epochs of learning.
	T = time.Now()
	for epoch, mG = 0, math.MaxFloat64; epoch < maxepochs; epoch++ {
		t = time.Now()
		// Switch to later learning parameters if appropriate.
		if epoch == maxepochs/2 {
			exag = 1.0
			l2 = 0.0
			momentum = 0.8
			uselr = lr
			// Reset the updates to 0 as learning parameters have changed.
			for object_ix = 0; object_ix < len(sh.objects); object_ix++ {
				for j = 0; j < sh.ndims; j++ {
					U0[object_ix][j] = 0.0
					U1[object_ix][j] = 0.0
				}
			}
		}
		// Create a channel for gradient information to be returned on.
		gradient_channel = make(chan GradientInfo, len(sh.objects))
		// Calculate gradient information for all objects using goroutines.
		for object_ix = 0; object_ix < len(sh.objects); object_ix++ {
			go sh.Gradient(object_ix, sh.locs[object_ix], knn, alpha, l2, exag, decay, gradient_channel)
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
				// Calculate the current update.
				U1[G[chanctr].object_ix][j] = (uselr * G[chanctr].gradient[j]) + (momentum * U0[G[chanctr].object_ix][j])
				// Apply the current update to the object's position.
				sh.locs[G[chanctr].object_ix][j] -= U1[G[chanctr].object_ix][j]
			}
		}
		mE /= float64(len(sh.objects))
		mG /= float64(len(sh.objects))
		// Report performance for epoch.
		fmt.Printf("Epoch %6d: mE=%.10e mG=%.10e (lr=%.3e, mom=%.3e, alpha=%.3e, l2=%.3e, exag=%.3e, odist=%.3e, epoch took %v; elapsed %v).\n", epoch, mE, mG, uselr, momentum, alpha, l2, exag, sh.OriginDistance(), time.Now().Sub(t), time.Now().Sub(T))
		// Update learning rate using "bold driver" method.
		if epoch > 0 {
			if mE < mE0 {
				// Error decreased, so increase learning rate a little and make current update last update for next epoch.
				uselr *= 1.05
				for object_ix = 0; object_ix < len(sh.objects); object_ix++ {
					for j = 0; j < sh.ndims; j++ {
						U0[object_ix][j] = U1[object_ix][j]
					}
				}
			} else {
				// Error didn't decrease, so undo the step that increased the error and reduce the learning rate.
				for object_ix = 0; object_ix < len(sh.objects); object_ix++ {
					for j = 0; j < sh.ndims; j++ {
						sh.locs[object_ix][j] += U1[object_ix][j]
					}
				}
				uselr *= 0.5
			}
		}
		// Update the previous error.
		mE0 = mE
	}
}

//
// Gradient method.
//

func (sh *Shoehorn) Gradient(object_ix int, object_loc []float64, knn int, alpha float64, l2 float64, exag float64, decay string, gradient_channel chan GradientInfo) {
	var (
		j, feature_ix                 int
		W, W2, E, p, q, Q, tmp1, tmp2, tmp3 float64
		G, T1, T2                     []float64
		N                             Weights
		n                             WeightPair
	)
	// Perform initializations.
	G = make([]float64, sh.ndims)
	T1 = make([]float64, sh.ndims)
	T2 = make([]float64, sh.ndims)
	// Get nearest neighbors and sum of weights.
	N = sh.Neighbors(object_ix, object_loc, knn, decay)
	// Iterate over features of object.
	for feature_ix, p = range sh.objects[object_ix].data {
		// Reset values of accumulating terms.
		Q = 0.0
		W = 0.0
		for j = 0; j < sh.ndims; j++ {
			T1[j], T2[j] = 0.0, 0.0
		}
		// Calculate exponential gradient terms.
		for _, n = range N {
			Q += n.weight * sh.objects[n.object_ix].data[feature_ix]
			W += n.weight
			switch {
				case decay == "exp":
					tmp1 = n.weight / n.distance
				case decay == "pow":
					tmp1 = (n.weight * n.weight) / n.distance
			}
			for j = 0; j < sh.ndims; j++ {
				tmp2 = tmp1 * (sh.locs[n.object_ix][j] - object_loc[j])
				T1[j] += tmp2 * sh.objects[n.object_ix].data[feature_ix]
				T2[j] += tmp2
			}
		}
		// Set the reconstruction probability.
		q = (alpha * p) + ((1.0 - alpha) * (Q / W))
		// Update gradient information.
		W2 = W * W
		tmp3 = ((alpha - 1.0) * p * exag / q)
		for j = 0; j < sh.ndims; j++ {
			G[j] += tmp3 * (((T1[j] * W) - (Q * T2[j])) / W2)
			// G[j] += ((alpha - 1.0) * p * exag / q) * (((T1[j] * W) - (Q * T2[j])) / W2)
		}
		// Update the error.
		E += (p * exag * (math.Log(p * exag) - math.Log(q)))
	}
	// Account for L2 penalty in error and gradient.
	for j = 0; j < sh.ndims; j++ {
		E += (l2 * object_loc[j] * object_loc[j])
		G[j] += (2.0 * l2 * object_loc[j])
	}
	// Return gradient information via the gradient channel.
	gradient_channel <- GradientInfo{object_ix: object_ix, gradient: G, error: E}
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

func (sh *Shoehorn) Neighbors(object_ix int, object_loc []float64, knn int, decay string) (N Weights) {
	N = sh.Weights(object_ix, object_loc, decay)
	if (knn > 0) && (knn < len(N)) {
		N = N[:knn]
	}
	return
}

func (sh *Shoehorn) Weights(object_ix int, object_loc []float64, decay string) (weights Weights) {
	var (
		w    WeightPair
		o, j int
		d    float64
	)
	// Set weight function depending on decay string.
	var weight_function = func(distance float64) float64 {
		return math.Exp(-distance)
	}
	if decay == "pow" {
		weight_function = func(distance float64) float64 {
			return math.Pow(1.0 + distance, -1.0)
		}
	}
	// Set distances and weights.
	for o = 0; o < len(sh.objects); o++ {
		// Calculate sum of squared distances.
		d = 0.0
		for j = 0; j < sh.ndims; j++ {
			d += math.Pow(object_loc[j]-sh.locs[o][j], 2.0)
		}
		// If the point isn't directly on top record it as a neighbor.
		if d > 0.0 {
			w = WeightPair{object_ix: o}
			w.distance = math.Pow(d, 0.5)
			w.weight = weight_function(w.distance)
			if w.weight > 0.0 {
				weights = append(weights, w)
			}
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
