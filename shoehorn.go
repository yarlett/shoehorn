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
	nobjs       int
	ftctr       int
	object_ixs  map[string]int
	feature_ixs map[string]int
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
			loc[i] = (rand.Float64() - 0.5) * 2.0
		}
		sh.L = append(sh.L, loc)
		sh.object_ixs[object_name] = len(sh.objects) - 1
		sh.nobjs += 1
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
// Learn method. Performs gradient-descent on object locations.
//

func (sh *Shoehorn) Learn(max_move float64, mom float64, numepochs int, alpha float64, l2_mode bool, l2_start float64, l2_end float64, output_prefix string) {
	var (
		epoch, o, j                                                                        int
		min_weight, tmpG, G, maxG, scale, mean_origin_distance, use_max_move, l2, l2_decay float64
		gradients                                                                          [][]float64
		T, t                                                                               time.Time
	)
	// Initialization.
	T = time.Now()
	min_weight = 0.0
	l2 = l2_start
	l2_decay = math.Pow((l2_end / l2_start), 1.0/float64(numepochs))

	// Perform initial scaling so impact of L2 punishment isn't overwhelming.
	_, e1, e2 := sh.Error(min_weight, alpha, l2)
	fmt.Printf("%v %v\n", e1, e2)
	for ; e2 > (e1/10.0); {
		for o = 0; o < sh.nobjs; o++ {
			for j = 0; j < sh.ndims; j++ {
				sh.L[o][j] *= 0.5
			}
		}
		_, e1, e2 = sh.Error(min_weight, alpha, l2)
		fmt.Printf("%v %v\n", e1, e2)
	}

	// Perform learning.
	for epoch = 0; epoch < numepochs; epoch++ {
		t = time.Now()
		// Get gradient for all objects.
		gradients = sh.Gradients(min_weight, alpha, l2)
		// Calculate mean magnitude of gradient vectors.
		G, maxG = 0.0, 0.0
		for o = 0; o < sh.nobjs; o++ {
			tmpG = sh.Magnitude(gradients[o])
			G += tmpG
			if tmpG > maxG {
				maxG = tmpG
			}
		}
		G /= float64(sh.nobjs)
		// Set maximum move.
		_, mean_origin_distance, _ = sh.DistanceInformation()
		if l2_mode {
			use_max_move = mean_origin_distance / 2.0
			if use_max_move > 0.1 {
				use_max_move = 0.1
			}
		} else {
			use_max_move = max_move
		}
		// Set the current updates and apply them.
		for o = 0; o < sh.nobjs; o++ {
			// scale = use_max_move / sh.Magnitude(gradients[o])
			scale = use_max_move / maxG
			for j = 0; j < sh.ndims; j++ {
				sh.L[o][j] -= (scale * gradients[o][j])
			}
		}
		// Report status.
		fmt.Printf("Epoch %6d: G=%.10e (use_max_move=%.4e mom=%.4e alpha=%.4e l2=%.4e odist=%.4e; epoch took %v; %v elapsed).\n", epoch+1, G, use_max_move, mom, alpha, l2, mean_origin_distance, time.Now().Sub(t), time.Now().Sub(T))
		// Write position of objects.
		if output_prefix != "" {
			sh.WriteLocations(fmt.Sprintf("%v_%v.csv", output_prefix, epoch+1))
		}
		// Reduce the l2 punishment parameter.
		if l2_mode {
			l2 *= l2_decay
			if l2 < l2_end {
				break
			}
		}
	}
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
	runtime.GOMAXPROCS(runtime.NumCPU())
	reconstruction_channel = make(chan ReconstructionInfo, len(sh.objects))
	for o = 0; o < sh.nobjs; o++ {
		go sh.ReconstructionWrapper(o, min_weight, reconstruction_channel)
	}
	R.WPS = make(map[int]map[int]float64)
	R.WS = make(map[int]float64)
	for o = 0; o < sh.nobjs; o++ {
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
// Error methods.
//

func (sh *Shoehorn) Error(min_weight float64, alpha float64, l2 float64) (E, E1, E2 float64) {
	var (
		R       ReconstructionSet
		o, j, f int
		p, q    float64
	)
	// Get the object reconstructions.
	R = sh.Reconstructions(min_weight)
	// Compute the error for each object.
	for o = 0; o < sh.nobjs; o++ {
		// Reconstruction error.
		for f, p = range sh.objects[o].data {
			q = (alpha * p) + ((1.0 - alpha) * (R.WPS[o][f] / R.WS[o]))
			E1 += (p * (math.Log(p) - math.Log(q)))
		}
		// Distance from origin punishment error.
		for j = 0; j < sh.ndims; j++ {
			E2 += l2 * math.Pow(sh.L[o][j], 2.0)
		}
	}
	E = E1 + E2
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
	// Compute gradient information.
	gradient_channel = make(chan GradientInfo, len(sh.objects))
	for o = 0; o < sh.nobjs; o++ {
		go sh.GradientWrapper(o, min_weight, alpha, l2, R, gradient_channel)
	}
	for o = 0; o < sh.nobjs; o++ {
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
	for o = 0; o < sh.nobjs; o++ {
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
	// Add distance from origin punishment gradient information.
	// for j = 0; j < sh.ndims; j++ {
	// 	gradient[j] += 2.0 * l2 * sh.L[object][j]
	// }

	foo := math.Pow(math.Pow(sh.Magnitude(sh.L[object]), 0.5), -0.5)
	for j = 0; j < sh.ndims; j++ {
		gradient[j] += l2 * sh.L[object][j] * foo
	}

	return
}

//
// Nearest neighbor methods.
//

func (sh *Shoehorn) Neighbors(object int, min_weight float64) (N Neighbors) {
	var (
		o, j             int
		distance, weight float64
	)
	for o = 0; o < sh.nobjs; o++ {
		if o != object {
			// Calculate distance.
			distance = 0.0
			for j = 0; j < sh.ndims; j++ {
				distance += math.Pow(sh.L[object][j]-sh.L[o][j], 2.0)
			}
			distance = math.Pow(distance, 0.5)
			weight = math.Exp(-distance)
			// If the point isn't 0 and the weight is above the minimum, add it as a neighbor.
			if weight >= min_weight {
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
	for id := 0; id < sh.nobjs; id++ {
		object_ids = append(object_ids, id)
	}
	return
}

// Ensures that the feature values for all objects have the same magnitude.
func (sh *Shoehorn) NormalizeObjects(metric float64) {
	for object := 0; object < sh.nobjs; object++ {
		mag := 0.0
		for _, v := range sh.objects[object].data {
			mag += math.Pow(v, metric)
		}
		mag = math.Pow(mag, 1.0/metric)
		for k, v := range sh.objects[object].data {
			sh.objects[object].Set(k, v/mag)
		}
	}
	return
}

// Returns information about the distance of points from the origin.
func (sh *Shoehorn) DistanceInformation() (min, mean, max float64) {
	min, mean, max = math.MaxFloat64, 0.0, 0.0
	for o := 0; o < sh.nobjs; o++ {
		distance := sh.Magnitude(sh.L[o])
		if distance < min {
			min = distance
		}
		mean += distance
		if distance > max {
			max = distance
		}
	}
	mean /= float64(sh.nobjs)
	return
}

// Creates new storage for position update.
func (sh *Shoehorn) GetObjectStore() (S [][]float64) {
	S = make([][]float64, len(sh.objects))
	for object := 0; object < sh.nobjs; object++ {
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
	// Seed the random number generator.
	rand.Seed(time.Now().Unix())
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
			_, seen := seenobjs[strvals[0]]
			if !seen {
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
	sh.NormalizeObjects(1.0)
	return
}
