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
			loc[i] = (rand.Float64() - 0.5) * 1e-2
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

func (sh *Shoehorn) LearnL2(l2_start float64, l2_end float64, numepochs int, alpha float64, output_prefix string) {
	var (
		epoch, o, j                                     int
		min_weight, l2, l2_mul, odist, step_size, scale float64
		G                                               [][]float64
		T, t                                            time.Time
	)
	// Initialization.
	T = time.Now()
	min_weight = 0.0
	l2 = l2_start
	l2_mul = math.Pow(l2_end/l2_start, 1.0/float64(numepochs-1))
	// Perform learning.
	for epoch = 0; epoch < numepochs; epoch++ {
		t = time.Now()
		// Set step size based on average distance from origin.
		_, odist, _ = sh.DistanceInformation()
		step_size = odist / 20.0
		if step_size > 0.5 {
			step_size = 0.5
		}
		// Get gradient for all objects and update positions.
		G = sh.Gradients(min_weight, alpha, l2)
		for o = 0; o < sh.nobjs; o++ {
			scale = step_size / sh.Magnitude(G[o])
			for j = 0; j < sh.ndims; j++ {
				sh.L[o][j] -= scale * G[o][j]
			}
		}
		// Report status.
		fmt.Printf("L2 Epoch %6d: l2=%.4e odist=%.4e step_size=%.4e alpha=%.4e; epoch took %v; %v elapsed).\n", epoch+1, l2, odist, step_size, alpha, time.Now().Sub(t), time.Now().Sub(T))
		// Write position of objects.
		if output_prefix != "" {
			sh.WriteLocations(fmt.Sprintf("%v_%v.csv", output_prefix, epoch+1))
		}
		// Rescale the object locations and update the l2 parameter.
		l2 *= l2_mul
	}
}

func (sh *Shoehorn) LearnRadius(radius1 float64, radius2 float64, numepochs int, alpha float64, output_prefix string) {
	var (
		epoch, o, j                           int
		min_weight, radius, radius_mul, scale float64
		G                                     [][]float64
		T, t                                  time.Time
	)
	// Initialization.
	T = time.Now()
	min_weight = 0.0
	radius = radius1
	sh.Rescale(radius)
	radius_mul = math.Pow(radius2/radius1, 1.0/float64(numepochs-1))
	// Perform learning.
	for epoch = 0; epoch < numepochs; epoch++ {
		t = time.Now()
		// Get gradient for all objects and update positions.
		G = sh.Gradients(min_weight, alpha, 0.0)
		for o = 0; o < sh.nobjs; o++ {
			scale = (radius / 20.0) / sh.Magnitude(G[o])
			for j = 0; j < sh.ndims; j++ {
				sh.L[o][j] -= scale * G[o][j]
			}
		}
		// Report status.
		fmt.Printf("Radius Epoch %6d: radius=%.4e alpha=%.4e; epoch took %v; %v elapsed).\n", epoch+1, radius, alpha, time.Now().Sub(t), time.Now().Sub(T))
		// Write position of objects.
		if output_prefix != "" {
			sh.WriteLocations(fmt.Sprintf("%v_%v.csv", output_prefix, epoch+1))
		}
		// Rescale the object locations and update the radius.
		radius *= radius_mul
		sh.Rescale(radius)
	}
}

func (sh *Shoehorn) LearnRprop(step_size float64, numepochs int, alpha float64, output_prefix string) {
	var (
		epoch, o, j          int
		min_weight, mgG, mgS float64
		G0, G1, S            [][]float64
		T, t                 time.Time
	)
	// Initialization.
	T = time.Now()
	min_weight = 0.0
	// Initialize Rprop parameters.
	G0 = sh.GetObjectStore()
	S = sh.GetObjectStore()
	for o = 0; o < sh.nobjs; o++ {
		for j = 0; j < sh.ndims; j++ {
			S[o][j] = step_size
		}
	}
	// Perform learning.
	for epoch = 0; epoch < numepochs; epoch++ {
		t = time.Now()
		// Get gradient for all objects.
		G1 = sh.Gradients(min_weight, alpha, 0.0)
		// Update positions using Rprop algorithm and update gradients.
		S = sh.Rprop(S, G0, G1, 0.1)
		G0 = G1
		// Compute magnitude of gradient and step sizes.
		mgG = 0.0
		mgS = 0.0
		for o = 0; o < sh.nobjs; o++ {
			mgG += sh.Magnitude(G1[o])
			mgS += sh.Magnitude(S[o])
		}
		mgG /= float64(sh.nobjs)
		mgS /= float64(sh.nobjs)
		// Report status.
		fmt.Printf("Rprop Epoch %6d: |G|=%.10e |S|=%.10e (alpha=%.4e; epoch took %v; %v elapsed).\n", epoch+1, mgG, mgS, alpha, time.Now().Sub(t), time.Now().Sub(T))
		// Write position of objects.
		if output_prefix != "" {
			sh.WriteLocations(fmt.Sprintf("%v_%v.csv", output_prefix, epoch+1))
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

func (sh *Shoehorn) Error(min_weight float64, alpha float64) (E float64) {
	var (
		R    ReconstructionSet
		o, f int
		p, q float64
	)
	// Get the object reconstructions.
	R = sh.Reconstructions(min_weight)
	// Compute the error for each object.
	for o = 0; o < sh.nobjs; o++ {
		// Reconstruction error.
		for f, p = range sh.objects[o].data {
			q = (alpha * p) + ((1.0 - alpha) * (R.WPS[o][f] / R.WS[o]))
			E += (p * (math.Log(p) - math.Log(q)))
		}
	}
	return
}

//
// Gradient methods.
//

func (sh *Shoehorn) Gradients(min_weight float64, alpha float64, l2 float64) (G [][]float64) {
	var (
		o                int
		R                ReconstructionSet
		gi               GradientInfo
		gradient_channel chan GradientInfo
	)
	G = make([][]float64, sh.nobjs)
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
		G[gi.object] = gi.gradient
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
	// Add L2 punishment.
	ssq := 0.0
	for j = 0; j < sh.ndims; j++ {
		ssq += math.Pow(sh.L[object][j], 2.0)
	}
	ssq = math.Pow(ssq, -0.5)
	for j = 0; j < sh.ndims; j++ {
		gradient[j] += l2 * ssq * sh.L[object][j]
	}
	// for j = 0; j < sh.ndims; j++ {
	// 	gradient[j] += l2 * math.Pow(sh.L[object][j], 2.0)
	// }
	return
}

func (sh *Shoehorn) Rprop(S, G0, G1 [][]float64, step_size_cap float64) [][]float64 {
	var (
		o, j             int
		gradient_product float64
	)
	for o = 0; o < sh.nobjs; o++ {
		for j = 0; j < sh.ndims; j++ {
			// Update the step size (consistent gradient directions get a boost, inconsistent directions get reduced).
			gradient_product = G0[o][j] * G1[o][j]
			if gradient_product > 0.0 {
				S[o][j] *= 1.1
				if S[o][j] > step_size_cap {
					S[o][j] = step_size_cap
				}
			} else if gradient_product < 0.0 {
				S[o][j] *= 0.5
			}
			// Update the position based on the sign of its magnitude and the learned step size (RProp doesn't use gradient magnitudes).
			if G1[o][j] > 0.0 {
				sh.L[o][j] -= S[o][j]
			} else {
				sh.L[o][j] += S[o][j]
			}
		}
	}
	return S
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

func (sh *Shoehorn) Rescale(radius float64) {
	var (
		o, j int
	)
	// Calculate centroid and maximum distance from origin.
	centroid := make([]float64, sh.ndims)
	max_distance := 0.0
	for o = 0; o < sh.nobjs; o++ {
		for j = 0; j < sh.ndims; j++ {
			centroid[j] += sh.L[o][j]
		}
		distance := sh.Magnitude(sh.L[o])
		if distance > max_distance {
			max_distance = distance
		}
	}
	for j = 0; j < sh.ndims; j++ {
		centroid[j] /= float64(sh.nobjs)
	}
	// Recenter and rescale each location.
	for o = 0; o < sh.nobjs; o++ {
		for j = 0; j < sh.ndims; j++ {
			sh.L[o][j] = (sh.L[o][j] - centroid[j]) * (radius / max_distance)
		}
	}
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
