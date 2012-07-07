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
// Initial positions.
//

func (sh *Shoehorn) InitialPositions(num_categories int, alpha float64) {
	var (
		o, j, c, current_category, epoch           int
		delta, remove_delta, add_delta, best_delta float64
		inds                                       []int
		deltas, cluster_center                     []float64
		categories                                 []map[int]bool
		t                                          time.Time
	)
	// Initialize storage for error deltas.
	deltas = make([]float64, num_categories)
	// Initialize the categories.
	categories = make([]map[int]bool, num_categories)
	for c = 0; c < len(categories); c++ {
		categories[c] = make(map[int]bool)
	}
	// Randomly assign objects to the categories.
	for o = 0; o < len(sh.objects); o++ {
		categories[rand.Intn(num_categories)][o] = true
	}
	fmt.Printf("Initial Positioning Random Assignment: E=%e.\n", sh.GlobalCategoryError(categories, alpha))
	// Reassign the objects to categories in order to improve the coherency of the categories.
	for epoch = 0; epoch < 10; epoch++ {
		t = time.Now()
		for o = 0; o < len(sh.objects); o++ {
			// Find the current category the object is assigned to.
			for current_category = 0; current_category < len(categories); current_category++ {
				if categories[current_category][o] {
					break
				}
			}
			// Get the error delta for removing the object from its current category.
			delta = sh.CategoryError(categories[current_category], alpha)
			delete(categories[current_category], o)
			remove_delta = sh.CategoryError(categories[current_category], alpha) - delta
			// Calculate error delta for adding the object t each category.
			for c = 0; c < len(categories); c++ {
				// If object is already in the category then the delta is 0.
				if c == current_category {
					deltas[c] = 0.0
				// Otherwise compute the delta associated with removing object from its current category and putting it in c.
				} else {
					delta = sh.CategoryError(categories[c], alpha)
					categories[c][o] = true
					add_delta = sh.CategoryError(categories[c], alpha) - delta
					delete(categories[c], o)
					deltas[c] = remove_delta + add_delta
				}
			}
			// Identify the category whose delta is lowest (include ties if they occur).
			best_delta = math.MaxFloat64
			for _, delta = range deltas {
				if delta < best_delta {
					best_delta = delta
				}
			}
			inds = nil
			for c, delta = range deltas {
				if delta == best_delta {
					inds = append(inds, c)
				}
			}
			// Randomly assign the object to any categories tied for biggest reduction delta.
			categories[inds[rand.Intn(len(inds))]][o] = true
		}
		// Assess the category goodnesses and report.
		fmt.Printf("Initial Positioning Epoch %d: E=%e (took %v).\n", epoch, sh.GlobalCategoryError(categories, alpha), time.Now().Sub(t))
	}
	// Assign each category to its own cluster.
	cluster_center = make([]float64, sh.ndims)
	for c = 0; c < len(categories); c++ {
		// Set cluster center.
		for j = 0; j < sh.ndims; j++ {
			cluster_center[j] = rand.NormFloat64() * 5.0
		}
		// Assign points around this cluster.
		for o, _ = range categories[c] {
			for j = 0; j < sh.ndims; j++ {
				sh.L[o][j] = cluster_center[j] + (rand.NormFloat64() * 0.3)
			}
		}
	}
}

func (sh *Shoehorn) GlobalCategoryError(categories []map[int]bool, alpha float64) (E float64) {
	for c := 0; c < len(categories); c++ {
		E += sh.CategoryError(categories[c], alpha)
	}
	return
}

func (sh *Shoehorn) CategoryError(category map[int]bool, alpha float64) (E float64) {
	var (
		o1, o2 int
		N      float64
	)
	for o1, _ = range category {
		for o2, _ = range category {
			if o2 != o1 {
				E += sh.objects[o1].KLDivergence(sh.objects[o2], alpha)
				N += 1.0
			}
		}
	}
	if N > 0.0 {
		E /= N
	}
	return
}

//
// Learning method.
//

func (sh *Shoehorn) Learn(lr float64, mom float64, l2 float64, numepochs int, alpha float64) {
	var (
		epoch, o, j, tries, maxtries int
		min_weight, Elst, Ecur, G    float64
		gradients, U                 [][]float64
		T, t                         time.Time
	)
	// Start timing.
	T = time.Now()
	// Initialization.
	min_weight = 0.0
	maxtries = 10
	U = sh.GetObjectStore()
	// Perform learning.
	for Elst, epoch = math.MaxFloat64, 0; (epoch < numepochs) && (lr > 1e-10); epoch++ {
		t = time.Now()
		// Get gradient for all objects.
		gradients = sh.Gradients(min_weight, alpha, l2)
		// Calculate magnitude of gradient vectors.
		G = 0.0
		for o = 0; o < len(sh.objects); o++ {
			G += sh.Magnitude(gradients[o])
		}
		G /= float64(len(sh.objects))
		// Update positions of objects.
		for tries = 0; tries < maxtries; tries++ {
			// Update position of each object.
			for o = 0; o < len(sh.objects); o++ {
				for j = 0; j < sh.ndims; j++ {
					sh.L[o][j] -= (lr * gradients[o][j]) + (mom * U[o][j])
				}
			}
			// Compute error.
			Ecur = sh.Error(min_weight, alpha, l2) / float64(len(sh.objects))
			// Perform actions depending on whether error was reduced or not.
			if Ecur < Elst {
				// Set update vectors.
				for o = 0; o < len(sh.objects); o++ {
					for j = 0; j < sh.ndims; j++ {
						U[o][j] = (lr * gradients[o][j]) + (mom * U[o][j])
					}
				}
				// Update the error and break out of the loop.
				Elst = Ecur
				break
			} else {
				// Unwind the changes and reduce the learning rate.
				for o = 0; o < len(sh.objects); o++ {
					for j = 0; j < sh.ndims; j++ {
						sh.L[o][j] += (lr * gradients[o][j]) + (mom * U[o][j])
					}
				}
				lr *= 0.5
			}
		}
		// Report status.
		fmt.Printf("Epoch %6d (%d tries): E=%.10e G=%.10e (lr=%.4e mom=%.4e alpha=%.4e l2=%.4e odist=%.4e; epoch took %v; %v elapsed).\n", epoch+1, tries+1, Ecur, G, lr, mom, alpha, l2, sh.OriginDistance(), time.Now().Sub(t), time.Now().Sub(T))
	}
}

//
// Error methods.
//

func (sh *Shoehorn) Error(min_weight float64, alpha float64, l2 float64) (E float64) {
	var (
		object, j, feature int
		WP, W, p           float64
		N                  Neighbors
		n                  Neighbor
	)
	for object = 0; object < len(sh.objects); object++ {
		// Calculate nearest neighbors.
		N = sh.Neighbors(object, min_weight)
		W = 0.0
		for _, n = range N {
			W += n.weight
		}
		// Iterate over the features of the object.
		for feature, p = range sh.objects[object].data {
			// Calculate the weighted sum for the feature.
			WP = 0.0
			for _, n = range N {
				WP += n.weight * sh.objects[n.object].data[feature]
			}
			// Update the error.
			E += (p * (math.Log(p) - math.Log((alpha*p)+((1.0-alpha)*(WP/W)))))
		}
		// Add punishment for distance from origin.
		for j = 0; j < sh.ndims; j++ {
			E += l2 * sh.L[object][j] * sh.L[object][j]
		}
	}
	return
}

//
// Gradient methods.
//

func (sh *Shoehorn) Gradients(min_weight float64, alpha float64, l2 float64) (gradients [][]float64) {
	var (
		o                int
		R                ReconstructionSet
		gi               GradientInfo
		gradient_channel chan GradientInfo
	)
	gradients = make([][]float64, len(sh.objects))
	runtime.GOMAXPROCS(runtime.NumCPU())
	// Precompute reconstruction data.
	R = sh.Reconstructions(min_weight)
	// Compute nearest neighbor information.
	gradient_channel = make(chan GradientInfo, len(sh.objects))
	for o = 0; o < len(sh.objects); o++ {
		go sh.GradientWrapper(o, min_weight, alpha, l2, R, gradient_channel)
	}
	for o = 0; o < len(sh.objects); o++ {
		gi = <-gradient_channel
		gradients[gi.object] = gi.gradient
	}
	return
}

func (sh *Shoehorn) GradientWrapper(object int, min_weight float64, alpha float64, l2 float64, R ReconstructionSet, channel chan GradientInfo) {
	channel <- GradientInfo{object: object, gradient: sh.Gradient(object, min_weight, alpha, l2, R)}
	return
}

func (sh *Shoehorn) Gradient(object int, min_weight float64, alpha float64, l2 float64, R ReconstructionSet) (gradient []float64) {
	var (
		o, j, feature                         int
		distance, weight, p, tmp1, tmp2, tmp3 float64
		T1, T2                                []float64
		N                                     Neighbors
		n                                     Neighbor
	)
	gradient = make([]float64, sh.ndims)
	T1 = make([]float64, sh.ndims)
	T2 = make([]float64, sh.ndims)
	// Compute impact of object position on its own reconstruction error.
	N = sh.Neighbors(object, min_weight)
	for feature, p = range sh.objects[object].data {
		// Calculate the gradient terms.
		for j = 0; j < sh.ndims; j++ {
			T1[j], T2[j] = 0.0, 0.0
		}
		for _, n = range N {
			tmp1 = n.weight / n.distance
			for j = 0; j < sh.ndims; j++ {
				tmp2 = tmp1 * (sh.L[n.object][j] - sh.L[object][j])
				T1[j] += tmp2 * sh.objects[n.object].data[feature]
				T2[j] += tmp2
			}
		}
		// Update gradient information.
		tmp1 = (alpha - 1.0) * p / ((alpha * p) + ((1.0 - alpha) * (R.WPS[object][feature] / R.WS[object])))
		for j = 0; j < sh.ndims; j++ {
			gradient[j] += tmp1 * (((T1[j] * R.WS[object]) - (R.WPS[object][feature] * T2[j])) / (R.WS[object] * R.WS[object]))
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
			tmp1 = weight / distance
			// Iterate over features of object getting reconstructed.
			for feature, p = range sh.objects[o].data {
				// Update gradient information.
				tmp2 = (alpha - 1.0) * p / ((alpha * p) + ((1.0 - alpha) * (R.WPS[o][feature] / R.WS[o])))
				for j = 0; j < sh.ndims; j++ {
					tmp3 = tmp1 * (sh.L[o][j] - sh.L[object][j])
					gradient[j] += tmp2 * (((R.WS[o] * tmp3 * sh.objects[object].data[feature]) - (R.WPS[o][feature] * tmp3)) / (R.WS[o] * R.WS[o]))
				}
			}
		}
	}
	// Add punishment for distance from origin.
	for j = 0; j < sh.ndims; j++ {
		gradient[j] += 2.0 * l2 * sh.L[object][j]
	}
	return
}

//
// Reconstruction methods.
//

func (sh *Shoehorn) Reconstructions(min_weight float64) (R ReconstructionSet) {
	var (
		o                      int
		reconstruction_channel chan ReconstructionInfo
		ri                     ReconstructionInfo
	)
	reconstruction_channel = make(chan ReconstructionInfo, len(sh.objects))
	for o = 0; o < len(sh.objects); o++ {
		go sh.ReconstructionWrapper(o, min_weight, reconstruction_channel)
	}
	R.WPS = make(map[int]map[int]float64)
	R.WS = make(map[int]float64)
	for o = 0; o < len(sh.objects); o++ {
		ri = <-reconstruction_channel
		R.WPS[ri.object] = ri.WP
		R.WS[ri.object] = ri.W
	}
	return
}

func (sh *Shoehorn) ReconstructionWrapper(object int, min_weight float64, channel chan ReconstructionInfo) {
	var (
		WP map[int]float64
		W  float64
	)
	WP, W = sh.Reconstruction(object, min_weight)
	channel <- ReconstructionInfo{object: object, WP: WP, W: W}
	return
}

func (sh *Shoehorn) Reconstruction(object int, min_weight float64) (WP map[int]float64, W float64) {
	var (
		feature int
		p       float64
		n       Neighbor
	)
	WP = make(map[int]float64)
	for _, n = range sh.Neighbors(object, min_weight) {
		W += n.weight
		for feature, p = range sh.objects[n.object].data {
			WP[feature] += n.weight * p
		}
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

// func (sh *Shoehorn) RepositioningSearch(cycles int, min_weight int, alpha float64) {
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
// 			go sh.Reposition(object, min_weight, alpha, C)
// 		}
// 		for object = 0; object < len(sh.objects); object++ {
// 			<-C
// 		}
// 		close(C)
// 		fmt.Printf("Objects repositioned in %v.\n", time.Now().Sub(t))
// 	}
// }

// func (sh *Shoehorn) Reposition(object int, min_weight int, alpha float64, C chan bool) {
// 	var (
// 		o, j int
// 		internalC chan bool
// 		bestloc []float64
// 		bestE float64
// 	)
// 	internalC = make(chan bool, 1)
// 	bestloc = make([]float64, sh.ndims)
// 	// Get the initial error for the object.
// 	sh.Gradient(object, sh.L[object], min_weight, alpha, internalC)
// 	<-internalC
// 	for j = 0; j < sh.ndims; j++ {
// 		bestloc[j] = sh.L[object][j]
// 	}	
// 	bestE = sh.E[object]
// 	// Try all repositionings.
// 	for o = 0; o < len(sh.objects); o++ {
// 		if o != object {
// 			sh.Gradient(object, sh.L[o], min_weight, alpha, internalC)
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

func (sh *Shoehorn) Neighbors(object int, min_weight float64) (N Neighbors) {
	var (
		o, j             int
		distance, weight float64
	)
	for o = 0; o < len(sh.objects); o++ {
		if o != object {
			// Calculate distance.
			distance = 0.0
			for j = 0; j < sh.ndims; j++ {
				distance += math.Pow(sh.L[object][j]-sh.L[o][j], 2.0)
			}
			distance = math.Pow(distance, 0.5)
			weight = math.Exp(-distance)
			// If the point isn't 0 and the weight is above the minimum, add it as a neighbor.
			if (distance > 0.0) && (weight >= min_weight) {
				N = append(N, Neighbor{object: o, distance: distance, weight: weight})
			}
		}
	}
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
