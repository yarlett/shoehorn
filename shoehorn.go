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
	lr, mom     float64
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
			loc[i] = rand.NormFloat64() * 0.1
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

func (sh *Shoehorn) Learn(lr float64, mom float64, l2 float64, numepochs int, alpha float64) {
	var (
		knn, epoch, object_ix, j, tries, numtries int
		E0, E1, G1                                float64
		T, t                                      time.Time
	)
	// Start timing.
	T = time.Now()
	// Initialization.
	sh.lr = lr
	sh.mom = mom
	knn = len(sh.objects)
	numtries = 5
	sh.Ulst = sh.GetObjectStore()
	sh.Ucur = sh.GetObjectStore()
	// Iterate over epochs of learning.
	sh.Gradients(knn, alpha, l2)
	for epoch = 0; epoch < numepochs; epoch++ {
		t = time.Now()
		// Set the current error.
		E0, _ = sh.Error()
		// Update the positions until error is reduced or the max number of tries is exceeded.
		for tries, E1 = 0, math.MaxFloat64; (E1 > E0) && (tries < numtries); tries++ {
			// Apply the current update to object positions.
			for object_ix = 0; object_ix < len(sh.objects); object_ix++ {
				for j = 0; j < sh.ndims; j++ {
					sh.Ucur[object_ix][j] = (sh.lr * sh.G[object_ix][j]) + (sh.mom * sh.Ulst[object_ix][j])
					sh.L[object_ix][j] -= sh.Ucur[object_ix][j]
				}
			}
			// Calculate the revised error.
			sh.Gradients(knn, alpha, l2)
			E1, G1 = sh.Error()
			// If the error has been reduced then success.
			if E1 < E0 {
				// Increase the learning rate by a modest amount.
				sh.lr *= 1.01
				// Make the current update the last update (used as the momentum term in the next epoch).
				for object_ix = 0; object_ix < len(sh.objects); object_ix++ {
					for j = 0; j < sh.ndims; j++ {
						sh.Ulst[object_ix][j] = sh.Ucur[object_ix][j]
					}
				}
				// Otherwise we have failed to reduce error so try again.
			} else {
				// Undo the current update and zero the last update (to kill momentum).
				for object_ix = 0; object_ix < len(sh.objects); object_ix++ {
					for j = 0; j < sh.ndims; j++ {
						sh.L[object_ix][j] += sh.Ucur[object_ix][j]
						sh.Ulst[object_ix][j] = 0.0
					}
				}
				// Reduce the learning rate significantly.
				sh.lr *= 0.5
			}
		}
		fmt.Printf("Epoch %6d (%d tries): E=%.10e G=%.10e (lr=%.4e mom=%.4e alpha=%.4e l2=%.4e odist=%.4e; epoch took %v; %v elapsed).\n", epoch+1, tries, E1, G1, sh.lr, sh.mom, alpha, l2, sh.OriginDistance(), time.Now().Sub(t), time.Now().Sub(T))
	}
}

//
// Gradient methods.
//

func (sh *Shoehorn) Gradients(knn int, alpha float64, l2 float64) {
	var (
		C         chan bool
		object_ix int
	)
	C = make(chan bool, len(sh.objects))
	// Set the maximum number of threads to be used to the number of CPU cores available.
	runtime.GOMAXPROCS(runtime.NumCPU())
	// Initialize error and gradient storage.
	sh.E = make([]float64, len(sh.objects))
	sh.G = sh.GetObjectStore()
	// Calculate gradient information for all objects using goroutines.
	for object_ix = 0; object_ix < len(sh.objects); object_ix++ {
		go sh.Gradient(object_ix, knn, alpha, l2, C)
	}
	// Collect all the gradient information from the gradient channel.
	for object_ix = 0; object_ix < len(sh.objects); object_ix++ {
		<-C
	}
}

func (sh *Shoehorn) Gradient(object_ix int, knn int, alpha float64, l2 float64, C chan bool) {
	var (
		j, feature_ix                    int
		SW, SWP, p, q, tmp1, tmp2, odist float64
		T1, T2                           []float64
		N                                Weights
		n                                WeightPair
	)
	// Perform initializations.
	T1 = make([]float64, sh.ndims)
	T2 = make([]float64, sh.ndims)
	// Get nearest neighbors and sum of weights.
	N = sh.Neighbors(object_ix, knn)
	// Iterate over features of object.
	for feature_ix, p = range sh.objects[object_ix].data {
		// Reset values of accumulating terms.
		SWP = 0.0
		SW = 0.0
		for j = 0; j < sh.ndims; j++ {
			T1[j], T2[j] = 0.0, 0.0
		}
		// Calculate gradient terms.
		for _, n = range N {
			SWP += n.weight * sh.objects[n.object_ix].data[feature_ix]
			SW += n.weight
			for j = 0; j < sh.ndims; j++ {
				tmp2 = n.weight * (sh.L[n.object_ix][j] - sh.L[object_ix][j]) / n.distance
				T1[j] += tmp2 * sh.objects[n.object_ix].data[feature_ix]
				T2[j] += tmp2
			}
		}
		// Set the reconstruction probability.
		q = (alpha * p) + ((1.0 - alpha) * (SWP / SW))
		// Update the error.
		sh.E[object_ix] += (p * (math.Log(p) - math.Log(q)))
		// Update gradient information.
		tmp1 = (alpha - 1.0) * p / q
		for j = 0; j < sh.ndims; j++ {
			sh.G[object_ix][j] += tmp1 * (((T1[j] * SW) - (SWP * T2[j])) / (SW * SW))
		}
		// Update gradient information of neighbor objects.
		for _, n = range N {
			for j = 0; j < sh.ndims; j++ {
				tmp2 = n.weight * (sh.L[object_ix][j] - sh.L[n.object_ix][j]) / n.distance
				sh.G[n.object_ix][j] += tmp1 * (((tmp2 * sh.objects[n.object_ix].data[feature_ix] * SW) - (SWP * tmp2)) / (SW * SW))
			}
		}
	}
	// Handle L2 coefficient.
	odist = sh.Magnitude(sh.L[object_ix])
	sh.E[object_ix] += l2 * odist
	for j = 0; j < sh.ndims; j++ {
		sh.G[object_ix][j] += (l2 * sh.L[object_ix][j] / odist)
	}
	// Signal gradient computation is complete.
	C <- true
}

// func (sh *Shoehorn) Rescale(radius float64) {
// 	var (
// 		object_ix, j int
// 		d, maxdist   float64
// 		centroid     []float64
// 	)
// 	centroid = make([]float64, sh.ndims)
// 	// Find maximum distance and centroid.
// 	for object_ix := 0; object_ix < len(sh.objects); object_ix++ {
// 		// Update maximum distance.
// 		d = sh.Magnitude(sh.L[object_ix])
// 		if d > maxdist {
// 			maxdist = d
// 		}
// 		// Update centroid.
// 		for j = 0; j < sh.ndims; j++ {
// 			centroid[j] += sh.L[object_ix][j]
// 		}
// 	}
// 	for j = 0; j < sh.ndims; j++ {
// 		centroid[j] = centroid[j] / float64(len(sh.objects))
// 	}
// 	// Recenter and rescale.
// 	for object_ix = 0; object_ix < len(sh.objects); object_ix++ {
// 		for j = 0; j < sh.ndims; j++ {
// 			sh.L[object_ix][j] = (sh.L[object_ix][j] - centroid[j]) * (radius / maxdist)
// 		}
// 	}
// }

// //
// // Repositioning method.
// //

// func (sh *Shoehorn) RepositioningSearch(cycles int, knn int, alpha float64) {
// 	// Initialization.
// 	var (
// 		object_ix     int
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
// 		for object_ix = 0; object_ix < len(sh.objects); object_ix++ {
// 			go sh.Reposition(object_ix, knn, alpha, C)
// 		}
// 		for object_ix = 0; object_ix < len(sh.objects); object_ix++ {
// 			<-C
// 		}
// 		close(C)
// 		fmt.Printf("Objects repositioned in %v.\n", time.Now().Sub(t))
// 	}
// }

// func (sh *Shoehorn) Reposition(object_ix int, knn int, alpha float64, C chan bool) {
// 	var (
// 		o, j int
// 		internalC chan bool
// 		bestloc []float64
// 		bestE float64
// 	)
// 	internalC = make(chan bool, 1)
// 	bestloc = make([]float64, sh.ndims)
// 	// Get the initial error for the object.
// 	sh.Gradient(object_ix, sh.L[object_ix], knn, alpha, internalC)
// 	<-internalC
// 	for j = 0; j < sh.ndims; j++ {
// 		bestloc[j] = sh.L[object_ix][j]
// 	}	
// 	bestE = sh.E[object_ix]
// 	// Try all repositionings.
// 	for o = 0; o < len(sh.objects); o++ {
// 		if o != object_ix {
// 			sh.Gradient(object_ix, sh.L[o], knn, alpha, internalC)
// 			<-internalC
// 			if sh.E[object_ix] < bestE {
// 				for j = 0; j < sh.ndims; j++ {
// 					bestloc[j] = sh.L[o][j]
// 				}
// 				bestE = sh.E[object_ix]
// 			}
// 		}
// 	}
// 	close(internalC)
// 	// Move object to the best position.
// 	for j = 0; j < sh.ndims; j++ {
// 		sh.L[object_ix][j] = bestloc[j] + (rand.NormFloat64() * 0.01)
// 	}
// 	C <- true
// }

//
// Utility methods.
//

func (sh *Shoehorn) ObjectIDs() (object_ids []int) {
	for id := 0; id < len(sh.objects); id++ {
		object_ids = append(object_ids, id)
	}
	return
}

func (sh *Shoehorn) Neighbors(object_ix int, knn int) (N Weights) {
	N = sh.Weights(object_ix)
	if (knn > 0) && (knn < len(N)) {
		N = N[:knn]
	}
	return
}

func (sh *Shoehorn) Weights(object_ix int) (weights Weights) {
	var (
		W        WeightPair
		o, j     int
		distance float64
	)
	// Set distances and weights.
	for o = 0; o < len(sh.objects); o++ {
		if o != object_ix {
			// Calculate distance.
			distance = 0.0
			for j = 0; j < sh.ndims; j++ {
				distance += math.Pow(sh.L[object_ix][j]-sh.L[o][j], 2.0)
			}
			distance = math.Pow(distance, 0.5)
			// If the point isn't directly on top record it as a neighbor.
			W = WeightPair{object_ix: o, distance: distance, weight: math.Exp(-distance)}
			weights = append(weights, W)
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
func (sh *Shoehorn) GetObjectStore() (S [][]float64) {
	S = make([][]float64, len(sh.objects))
	for object_ix := 0; object_ix < len(sh.objects); object_ix++ {
		S[object_ix] = make([]float64, sh.ndims)
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

func (sh *Shoehorn) Error() (E, G float64) {
	for object_ix := 0; object_ix < len(sh.objects); object_ix++ {
		E += sh.E[object_ix]
		G += sh.Magnitude(sh.G[object_ix])
	}
	E /= float64(len(sh.objects))
	G /= float64(len(sh.objects))
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
