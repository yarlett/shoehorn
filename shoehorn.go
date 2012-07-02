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
	L           [][]float64
	lr, mom     float64
}

//
// Object loading methods.
//

func (sh *Shoehorn) Store(object_name string, feature_name string, value float64) {
	// Update object indices.
	_, obfound := sh.object_ixs[object_name]
	if !obfound {
		// Create new feature vector and location for the object if it hasn't been encountered before.
		sh.objects = append(sh.objects, &FeatureVector{data: make(map[int]float64)})
		loc := make([]float64, sh.ndims)
		for i := 0; i < sh.ndims; i++ {
			loc[i] = rand.NormFloat64() * 0.01
		}
		sh.L = append(sh.L, loc)
		sh.object_ixs[object_name] = len(sh.objects) - 1
	}
	// Update feature indices.
	_, ftfound := sh.feature_ixs[feature_name]
	if !ftfound {
		sh.feature_ixs[feature_name] = sh.ftctr
		sh.ftctr++
	}
	// Store the feature value for the object.
	object, _ := sh.object_ixs[object_name]
	feature, _ := sh.feature_ixs[feature_name]
	sh.objects[object].Set(feature, value)
}

//
// Learning method.
//

func (sh *Shoehorn) Learn(lr float64, mom float64, l2 float64, numepochs int, alpha float64) {
	var (
		knn, epoch, o, j int
		Elst, E, G       float64
		// gradient         []float64
		U    [][]float64
		T, t time.Time
	)
	// Start timing.
	T = time.Now()
	// Initialization.
	knn = len(sh.objects)
	U = sh.GetObjectStore()
	// Perform learning.
	Elst = sh.Error(knn, alpha, l2) / float64(len(sh.objects))
	for epoch = 0; epoch < numepochs; epoch++ {
		t = time.Now()
		// For each object compute gradient and update position.
		gradients := sh.Gradients(knn, alpha, l2)
		G = 0.0
		for o = 0; o < len(sh.objects); o++ {
			//gradient = sh.Gradient(o, knn, alpha, l2)
			for j = 0; j < sh.ndims; j++ {
				U[o][j] = (lr * gradients[o][j]) + (mom * U[o][j])
				//U[o][j] = (lr * gradient[j]) + (mom * U[o][j])
				sh.L[o][j] -= U[o][j]
			}
			G += sh.Magnitude(gradients[o])
			//G += sh.Magnitude(gradient)
		}
		G /= float64(len(sh.objects))
		E = sh.Error(knn, alpha, l2) / float64(len(sh.objects))
		// If error not reduced, unwind updates and reduce error rate.
		if E >= Elst {
			for o = 0; o < len(sh.objects); o++ {
				for j = 0; j < sh.ndims; j++ {
					sh.L[o][j] += U[o][j]
					U[o][j] = 0.0
				}
			}
			lr *= 0.5
			// Otherwise update last error and continue.
		} else {
			Elst = E
		}
		// Report status.
		fmt.Printf("Epoch %6d: E=%.10e G=%.10e (lr=%.4e mom=%.4e alpha=%.4e l2=%.4e odist=%.4e; epoch took %v; %v elapsed).\n", epoch+1, E, G, lr, mom, alpha, l2, sh.OriginDistance(), time.Now().Sub(t), time.Now().Sub(T))
	}
}

//
// Error and gradient methods.
//

func (sh *Shoehorn) Error(knn int, alpha float64, l2 float64) (E float64) {
	var (
		object, feature int
		WP, W, p        float64
		N               Weights
		n               WeightPair
	)
	for object = 0; object < len(sh.objects); object++ {
		// Calculate nearest neighbors.
		N = sh.Neighbors(object, knn)
		// Calculate the sum of weights.
		W = 0.0
		for _, n = range N {
			W += n.weight
		}
		// Iterate over the features of the object.
		for feature, p = range sh.objects[object].data {
			// Calculate gradient terms.
			WP = 0.0
			for _, n = range N {
				WP += n.weight * sh.objects[n.object].data[feature]
			}
			// Update the error.
			E += (p * (math.Log(p) - math.Log((alpha*p)+((1.0-alpha)*(WP/W)))))
		}
		// Handle L2 coefficient.
		E += l2 * sh.Magnitude(sh.L[object])
	}
	return
}

func (sh *Shoehorn) Gradients(knn int, alpha float64, l2 float64) (gradients [][]float64) {
	var (
		o       int
		gi      GradientInfo
		channel chan GradientInfo
	)
	gradients = make([][]float64, len(sh.objects))
	channel = make(chan GradientInfo, len(sh.objects))
	runtime.GOMAXPROCS(runtime.NumCPU())
	// Compute nearest neighbor information.
	for o = 0; o < len(sh.objects); o++ {
		go sh.GradientWrapper(o, knn, alpha, l2, channel)
	}
	for o = 0; o < len(sh.objects); o++ {
		gi = <-channel
		gradients[gi.object] = gi.gradient
	}
	return
}

func (sh *Shoehorn) GradientWrapper(object int, knn int, alpha float64, l2 float64, channel chan GradientInfo) {
	channel <- GradientInfo{object: object, gradient: sh.Gradient(object, knn, alpha, l2)}
	return
}

func (sh *Shoehorn) Gradient(object int, knn int, alpha float64, l2 float64) (gradient []float64) {
	var (
		o, j, feature                                         int
		distance, weight, p, WP, W, Q, pre1, pre2, top1, top2 float64
		T1, T2                                                []float64
		N                                                     Weights
		n                                                     WeightPair
	)
	gradient = make([]float64, sh.ndims)
	T1 = make([]float64, sh.ndims)
	T2 = make([]float64, sh.ndims)
	// Compute impact of object position on its own reconstruction error.
	N = sh.Neighbors(object, knn)
	W = 0.0
	for _, n = range N {
		W += n.weight
	}
	// Iterate over features of current object.
	for feature, p = range sh.objects[object].data {
		// Reset values of accumulating terms.
		WP = 0.0
		for j = 0; j < sh.ndims; j++ {
			T1[j], T2[j] = 0.0, 0.0
		}
		// Calculate gradient terms.
		for _, n = range N {
			WP += n.weight * sh.objects[n.object].data[feature]
			for j = 0; j < sh.ndims; j++ {
				pre2 = n.weight * (sh.L[n.object][j] - sh.L[object][j]) / n.distance
				T1[j] += pre2 * sh.objects[n.object].data[feature]
				T2[j] += pre2
			}
		}
		// Set the reconstruction probability.
		Q = (alpha * p) + ((1.0 - alpha) * (WP / W))
		// Update gradient information.
		pre1 = (alpha - 1.0) * p / Q
		for j = 0; j < sh.ndims; j++ {
			gradient[j] += pre1 * (((T1[j] * W) - (WP * T2[j])) / (W * W))
		}
	}
	// Compute impact of object position on reconstruction error of other objects.
	for o = 0; o < len(sh.objects); o++ {
		if o != object {
			// Calculate distance and weight between current object and object being reconstructed.
			distance = 0.0
			for j = 0; j < sh.ndims; j++ {
				distance += math.Pow(sh.L[object][j]-sh.L[o][j], 2.0)
			}
			distance = math.Pow(distance, 0.5)
			weight = math.Exp(-distance)
			pre1 = weight / distance
			// Calculate nearest neighbors.
			N = sh.Neighbors(o, knn)
			W = 0.0
			for _, n = range N {
				W += n.weight
			}
			// Iterate over features of object getting reconstructed.
			for feature, p = range sh.objects[o].data {
				// Calculate reconstruction terms.
				WP = 0.0
				for _, n = range N {
					WP += n.weight * sh.objects[n.object].data[feature]
				}
				// Calculate reconstruction.
				Q = (alpha * p) + ((1.0 - alpha) * (WP / W))
				// Update gradient information.
				pre2 = (alpha - 1.0) * p / Q
				for j = 0; j < sh.ndims; j++ {
					top1 = pre1 * (sh.L[o][j] - sh.L[object][j]) * sh.objects[object].data[feature]
					top2 = pre1 * (sh.L[o][j] - sh.L[object][j])
					gradient[j] += pre2 * (((W * top1) - (WP * top2)) / (W * W))
				}
			}
		}
	}
	// Add punishment for distance from origin.
	distance = sh.Magnitude(sh.L[object])
	for j = 0; j < sh.ndims; j++ {
		gradient[j] += (l2 * sh.L[object][j] / distance)
	}
	return
}

//
// Rescaling methods.
//

// func (sh *Shoehorn) Rescale(radius float64) {
// 	var (
// 		object, j int
// 		d, maxdist   float64
// 		centroid     []float64
// 	)
// 	centroid = make([]float64, sh.ndims)
// 	// Find maximum distance and centroid.
// 	for object := 0; object < len(sh.objects); object++ {
// 		// Update maximum distance.
// 		d = sh.Magnitude(sh.L[object])
// 		if d > maxdist {
// 			maxdist = d
// 		}
// 		// Update centroid.
// 		for j = 0; j < sh.ndims; j++ {
// 			centroid[j] += sh.L[object][j]
// 		}
// 	}
// 	for j = 0; j < sh.ndims; j++ {
// 		centroid[j] = centroid[j] / float64(len(sh.objects))
// 	}
// 	// Recenter and rescale.
// 	for object = 0; object < len(sh.objects); object++ {
// 		for j = 0; j < sh.ndims; j++ {
// 			sh.L[object][j] = (sh.L[object][j] - centroid[j]) * (radius / maxdist)
// 		}
// 	}
// }

// //
// // Repositioning method.
// //

// func (sh *Shoehorn) RepositioningSearch(cycles int, knn int, alpha float64) {
// 	// Initialization.
// 	var (
// 		object     int
// 		t             time.Time
// 		C             chan bool
// 	)
// 	// Set the maximum number of threads to be used to the number of CPU cores available.
// 	runtime.GOMAXPROCS(runtime.NumCPU())
// 	// Iterate through cycles.
// 	for c := 0; c < cycles; c++ {
// 		// Run repositioning search on each object.
// 		t = time.Now()
// 		C = make(chan bool, len(sh.objects))
// 		for object = 0; object < len(sh.objects); object++ {
// 			go sh.Reposition(object, knn, alpha, C)
// 		}
// 		for object = 0; object < len(sh.objects); object++ {
// 			<-C
// 		}
// 		close(C)
// 		fmt.Printf("Objects repositioned in %v.\n", time.Now().Sub(t))
// 	}
// }

// func (sh *Shoehorn) Reposition(object int, knn int, alpha float64, C chan bool) {
// 	var (
// 		o, j int
// 		internalC chan bool
// 		bestloc []float64
// 		bestE float64
// 	)
// 	internalC = make(chan bool, 1)
// 	bestloc = make([]float64, sh.ndims)
// 	// Get the initial error for the object.
// 	sh.Gradient(object, sh.L[object], knn, alpha, internalC)
// 	<-internalC
// 	for j = 0; j < sh.ndims; j++ {
// 		bestloc[j] = sh.L[object][j]
// 	}	
// 	bestE = sh.E[object]
// 	// Try all repositionings.
// 	for o = 0; o < len(sh.objects); o++ {
// 		if o != object {
// 			sh.Gradient(object, sh.L[o], knn, alpha, internalC)
// 			<-internalC
// 			if sh.E[object] < bestE {
// 				for j = 0; j < sh.ndims; j++ {
// 					bestloc[j] = sh.L[o][j]
// 				}
// 				bestE = sh.E[object]
// 			}
// 		}
// 	}
// 	close(internalC)
// 	// Move object to the best position.
// 	for j = 0; j < sh.ndims; j++ {
// 		sh.L[object][j] = bestloc[j] + (rand.NormFloat64() * 0.01)
// 	}
// 	C <- true
// }

//
// Nearest neighbor methods.
//

func (sh *Shoehorn) Neighbors(object int, knn int) (N Weights) {
	N = sh.Weights(object)
	if (knn > 0) && (knn < len(N)) {
		N = N[:knn]
	}
	return
}

func (sh *Shoehorn) Weights(object int) (weights Weights) {
	var (
		W        WeightPair
		o, j     int
		distance float64
	)
	// Set distances and weights.
	for o = 0; o < len(sh.objects); o++ {
		if o != object {
			// Calculate distance.
			distance = 0.0
			for j = 0; j < sh.ndims; j++ {
				distance += math.Pow(sh.L[object][j]-sh.L[o][j], 2.0)
			}
			distance = math.Pow(distance, 0.5)
			// If the point isn't directly on top record it as a neighbor.
			W = WeightPair{object: o, distance: distance, weight: math.Exp(-distance)}
			weights = append(weights, W)
		}
	}
	sort.Sort(weights)
	return
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

// Ensures that the feature values for all objects sum to 1.
func (sh *Shoehorn) NormalizeObjectSums() {
	for object := 0; object < len(sh.objects); object++ {
		sum := sh.objects[object].Sum()
		for k, v := range sh.objects[object].data {
			sh.objects[object].Set(k, v/sum)
		}
	}
	return
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
func (sh *Shoehorn) GetObjectStore() (S [][]float64) {
	S = make([][]float64, len(sh.objects))
	for object := 0; object < len(sh.objects); object++ {
		S[object] = make([]float64, sh.ndims)
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

// Writes the current locations of objects to a file.
func (sh *Shoehorn) WriteLocations(path string) {
	// Initialize the output file.
	of, err := os.Create(path)
	if err != nil {
		log.Fatal(err)
	}
	defer of.Close()
	// Write object locations to file.
	for object_name, object := range sh.object_ixs {
		line := make([]string, 0)
		line = append(line, object_name)
		for i := 0; i < sh.ndims; i++ {
			line = append(line, fmt.Sprintf("%v", sh.L[object][i]))
		}
		of.Write([]byte(fmt.Sprintf("%v\n", strings.Join(line, ","))))
	}
}

//
// Constructs a DataSet from a CSV file of {object_name, feature_name, feature_value} triples.
//

func NewShoehorn(filename string, ndims int, downsample float64) (sh *Shoehorn) {
	var (
		bfr                  *bufio.Reader
		seenobjs, sampleobjs map[string]bool
	)
	seenobjs = make(map[string]bool)
	sampleobjs = make(map[string]bool)
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
			// If the object has not been seen before, decide whether to include it.
			if !seenobjs[strvals[0]] {
				if rand.Float64() < downsample {
					sampleobjs[strvals[0]] = true
				}
			}
			seenobjs[strvals[0]] = true
			// Store the data in the data set if the object is to be sampled.
			if sampleobjs[strvals[0]] {
				sh.Store(strvals[0], strvals[1], value_float)
			}
		}
		// Read from the file for the next iteration.
		line, isprefix, err = bfr.ReadLine()
	}
	// Normalize each vector so they sum to 1.
	sh.NormalizeObjectSums()
	return
}
