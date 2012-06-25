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
	object_ixs  map[string]int
	feature_ixs map[string]int
	ftctr       int
	objects     []*FeatureVector
	E           []float64
	L           [][]float64
	G           [][]float64
	Ulst        [][]float64
	Ucur        [][]float64
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
		sh.L = append(sh.L, loc)
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

func (sh *Shoehorn) Learn(lr float64, mom float64, maxepochs int, alpha float64, exag float64, l2 float64, decay string, error string) {
	var (
		knn, epoch, object_ix, j int
		mElst, mE, mG            float64
		T, t                     time.Time
	)
	// Initialization.
	mElst = math.MaxFloat64
	knn = len(sh.objects)
	sh.E = make([]float64, len(sh.objects))
	sh.G = sh.GetObjectStore()
	sh.Ulst = sh.GetObjectStore()
	sh.Ucur = sh.GetObjectStore()
	// Set the maximum number of threads to be used to the number of CPU cores available.
	numprocs := runtime.NumCPU()
	runtime.GOMAXPROCS(numprocs)
	// Iterate over epochs of learning.
	T = time.Now()
	for epoch, mG = 0, math.MaxFloat64; epoch < maxepochs; epoch++ {
		t = time.Now()
		// Compute and set all gradients.
		sh.Gradients(knn, alpha, exag, l2, decay, error)
		// Update positions based on gradient information.
		mE = 0.0
		mG = 0.0
		for object_ix = 0; object_ix < len(sh.L); object_ix++ {
			mE += sh.E[object_ix]
			mG += sh.Magnitude(sh.G[object_ix])
			for j = 0; j < sh.ndims; j++ {
				sh.Ucur[object_ix][j] = (lr * sh.G[object_ix][j]) + (mom * sh.Ulst[object_ix][j])
				sh.L[object_ix][j] -= sh.Ucur[object_ix][j]
			}
		}
		mE /= float64(len(sh.objects))
		mG /= float64(len(sh.objects))
		// Report performance for epoch.
		fmt.Printf("Epoch %6d: mE=%.10e mG=%.10e (lr=%.3e, mom=%.3e, alpha=%.3e, exag=%.3e, l2=%.3e, odist=%.3e, epoch took %v; elapsed %v).\n", epoch, mE, mG, lr, mom, alpha, exag, l2, sh.OriginDistance(), time.Now().Sub(t), time.Now().Sub(T))
		// Update learning rate using "bold driver" method.
		if epoch > 0 {
			if mE < mElst {
				// Error decreased, so increase learning rate a little and make current update last update for next epoch.
				lr *= 1.05
				sh.Ulst = sh.Ucur
				mElst = mE
			} else {
				// Error didn't decrease, so undo the step that increased the error, zero the last update to kill momentum, and reduce the learning rate.
				for object_ix = 0; object_ix < len(sh.objects); object_ix++ {
					for j = 0; j < sh.ndims; j++ {
						sh.L[object_ix][j] += sh.Ucur[object_ix][j]
					}
				}
				sh.Ulst = sh.GetObjectStore()
				lr *= 0.5
			}
		}
	}
}

//
// Gradient methods.
//

func (sh *Shoehorn) Gradients(knn int, alpha float64, exag float64, l2 float64, decay string, error string) {
	// Create a channel for gradient information to be returned on.
	C := make(chan int, len(sh.objects))
	// Calculate gradient information for all objects using goroutines.
	for object_ix := 0; object_ix < len(sh.objects); object_ix++ {
		go sh.Gradient(object_ix, knn, alpha, exag, l2, decay, error, C)
	}
	// Collect all the gradient information from the gradient channel.
	for object_ix := 0; object_ix < len(sh.objects); object_ix++ {
		<-C
	}
	close(C)
}

func (sh *Shoehorn) Gradient(object_ix int, knn int, alpha float64, exag float64, l2 float64, decay string, error string, C chan int) {
	var (
		j, feature_ix          int
		W, p, q, Q, g1, g2, e1 float64
		T1, T2                 []float64
		N                      Weights
		n                      WeightPair
		decay_term             func(float64, float64) float64
		error_term             func(float64, float64, float64, float64, float64) float64
		error_func             func(float64, float64, float64) float64
	)
	// Define decay term function based on decay type.
	switch {
	case decay == "exp":
		decay_term = func(distance, weight float64) float64 {
			return weight / distance
		}
	case decay == "pow":
		decay_term = func(distance, weight float64) float64 {
			return (weight * weight) / distance
		}
	}
	// Define error functions based on error type.
	switch {
	case error == "kl":
		error_term = func(p, Q, W, alpha, exag float64) float64 {
			return ((alpha - 1.0) * p * exag) / ((alpha * p) + ((1.0 - alpha) * (Q / W)))
		}
		error_func = func(p, q, exag float64) float64 {
			return p * exag * (math.Log(p*exag) - math.Log(q))
		}
	case error == "l2":
		error_term = func(p, Q, W, alpha, exag float64) float64 {
			return 2.0 * (1.0 - alpha) * (((alpha - exag) * p) + ((1.0 - alpha) * (Q / W)))
		}
		error_func = func(p, q, exag float64) float64 {
			return math.Pow(q-(p*exag), 2.0)
		}
	}
	// Perform initializations.
	sh.E[object_ix] = 0.0
	for j = 0; j < sh.ndims; j++ {
		sh.G[object_ix][j] = 0.0
	}
	T1 = make([]float64, sh.ndims)
	T2 = make([]float64, sh.ndims)
	// Get nearest neighbors and sum of weights.
	N = sh.Neighbors(object_ix, knn, decay)
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
			g1 = decay_term(n.distance, n.weight)
			for j = 0; j < sh.ndims; j++ {
				g2 = g1 * (sh.L[n.object_ix][j] - sh.L[object_ix][j])
				T1[j] += g2 * sh.objects[n.object_ix].data[feature_ix]
				T2[j] += g2
			}
		}
		// Set the reconstruction probability.
		q = (alpha * p) + ((1.0 - alpha) * (Q / W))
		// Update the error.
		sh.E[object_ix] += error_func(p, q, exag)
		// Update gradient information.
		e1 = error_term(p, Q, W, alpha, exag)
		for j = 0; j < sh.ndims; j++ {
			sh.G[object_ix][j] += e1 * (((T1[j] * W) - (Q * T2[j])) / (W * W))
		}
	}
	// Account for L2 penalty in error and gradient.
	for j = 0; j < sh.ndims; j++ {
		sh.E[object_ix] += (l2 * sh.L[object_ix][j] * sh.L[object_ix][j])
		sh.G[object_ix][j] += (2.0 * l2 * sh.L[object_ix][j])
	}
	// Signal gradient computation is complete.
	C <- 1
}

//
// Repositioning method.
//

func (sh *Shoehorn) Reposition(knn int, alpha float64, exag float64, l2 float64, decay string, error string) {
	// Initialization.
	var (
		o, o1, o2, j  int
		bestE, E      float64
		best_location []float64
		t time.Time
	)
	best_location = make([]float64, sh.ndims)
	// Try repositioning each object.
	for o1 = 0; o1 < len(sh.objects); o1++ {
		t = time.Now()
		// Initialize the best location and error.
		for j = 0; j < sh.ndims; j++ {
			best_location[j] = sh.L[o1][j]
		}
		sh.Gradients(knn, alpha, exag, l2, decay, error)
		bestE = 0.0
		for o = 0; o < len(sh.objects); o++ {
			bestE += sh.E[o]
		}
		bestE /= float64(len(sh.objects))
		fmt.Printf("Repositioning object %d (baseline E=%v).\n", o1, bestE)
		// Try relocating o1 to position of every other object.
		for o2 = 0; o2 < len(sh.objects); o2++ {
			if o2 != o1 {
				// Reposition o1 to be at o2's location.
				for j = 0; j < sh.ndims; j++ {
					sh.L[o1][j] = sh.L[o2][j]
				}
				// Get comparison error.
				sh.Gradients(knn, alpha, exag, l2, decay, error)
				E = 0.0
				for o = 0; o < len(sh.objects); o++ {
					E += sh.E[o]
				}
				E /= float64(len(sh.objects))
				// Update best results.
				if E < bestE {
					fmt.Printf("  Moving object %d to %d yields E=%v.\n", o1, o2, E)
					for j = 0; j < sh.ndims; j++ {
						best_location[j] = sh.L[o1][j]
					}
					bestE = E
				}
			}
		}
		// Relocate o1 to the best location found.
		for j = 0; j < sh.ndims; j++ {
			sh.L[o1][j] = best_location[j]
		}
		fmt.Printf("Object %d repositioned in %v with E=%v.\n", o1, time.Now().Sub(t), bestE)
	}
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

func (sh *Shoehorn) Neighbors(object_ix int, knn int, decay string) (N Weights) {
	N = sh.Weights(object_ix, decay)
	if (knn > 0) && (knn < len(N)) {
		N = N[:knn]
	}
	return
}

func (sh *Shoehorn) Weights(object_ix int, decay string) (weights Weights) {
	var (
		W               WeightPair
		o, j            int
		d               float64
		weight_function func(float64) float64
	)
	// Set weight function depending on decay string.
	switch {
	case decay == "exp":
		weight_function = func(distance float64) float64 {
			return math.Exp(-distance)
		}
	case decay == "pow":
		weight_function = func(distance float64) float64 {
			return math.Pow(1.0+distance, -1.0)
		}
	}
	// Set distances and weights.
	for o = 0; o < len(sh.objects); o++ {
		if o != object_ix {
			// Calculate sum of squared distances.
			d = 0.0
			for j = 0; j < sh.ndims; j++ {
				d += math.Pow(sh.L[object_ix][j]-sh.L[o][j], 2.0)
			}
			// If the point isn't directly on top record it as a neighbor.
			W = WeightPair{object_ix: o}
			W.distance = math.Pow(d, 0.5)
			W.weight = weight_function(W.distance)
			if W.weight > 0.0 {
				weights = append(weights, W)
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
			line = append(line, fmt.Sprintf("%v", sh.L[object_ix][i]))
		}
		of.Write([]byte(fmt.Sprintf("%v\n", strings.Join(line, ","))))
	}
}

// Returns the average distance of objects from the origin.
func (sh *Shoehorn) OriginDistance() (dist float64) {
	var n float64
	for o := 0; o < len(sh.objects); o++ {
		dist += sh.Magnitude(sh.L[o])
		n += 1.0
	}
	dist = dist / n
	return
}

// Creates new storage for position update.
func (sh *Shoehorn) GetObjectStore() (U [][]float64) {
	U = make([][]float64, len(sh.objects))
	for object_ix := 0; object_ix < len(sh.objects); object_ix++ {
		U[object_ix] = make([]float64, sh.ndims)
	}
	return
}

func (sh *Shoehorn) Magnitude(V []float64) (mag float64) {
	for i := 0; i < len(V); i++ {
		mag += V[i] * V[i]
	}
	mag = math.Pow(mag, 0.5)
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
