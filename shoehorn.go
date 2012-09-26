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
	nd   int
	no   int
	nf   int
	oixs map[string]int
	fixs map[string]int
	O    [][]float64
	L    [][]float64
}

//
// Shoehorn creation method.
//

func (sh *Shoehorn) Create(S [][]string, ndims int) {
	// Initializations.
	sh.oixs = make(map[string]int, 0)
	sh.fixs = make(map[string]int, 0)
	// Construct object and feature indices.
	octr := 0
	fctr := 0
	for _, strings := range S {
		// Update object index.
		_, obfound := sh.oixs[strings[0]]
		if !obfound {
			sh.oixs[strings[0]] = octr
			octr++
		}
		// Update feature index.
		_, ftfound := sh.fixs[strings[1]]
		if !ftfound {
			sh.fixs[strings[1]] = fctr
			fctr++
		}
	}
	// Set numbers of things.
	sh.nd = ndims
	sh.no = len(sh.oixs)
	sh.nf = len(sh.fixs)
	// Create objects.
	sh.O = make([][]float64, sh.no)
	for o := 0; o < sh.no; o++ {
		sh.O[o] = make([]float64, sh.nf)
	}
	// Populate object values.
	for _, strings := range S {
		o := sh.oixs[strings[0]]
		f := sh.fixs[strings[1]]
		value, _ := strconv.ParseFloat(strings[2], 64)
		sh.O[o][f] = value
	}
	// Set locations.
	sh.L = make([][]float64, sh.no)
	for o := 0; o < sh.no; o++ {
		sh.L[o] = make([]float64, sh.nd)
		for j := 0; j < sh.nd; j++ {
			sh.L[o][j] = (rand.Float64() - 0.5) * 1e-2
		}
	}
}

//
// Learn methods. Performs gradient-descent on object locations.
//

func (sh *Shoehorn) LearnGradientDescent(step_size float64, l2 float64, alpha float64, numepochs int, output_prefix string) {
	/*
		Uses gradient descent to find the best location for objects.
	*/
	var (
		epoch, o           int
		min_weight, mE, mD float64
		G                  []GradientInfo
		T, t               time.Time
	)
	// Initialization.
	T = time.Now()
	min_weight = 0.0
	// Perform learning.
	for epoch = 0; epoch < numepochs; epoch++ {
		t = time.Now()
		// Get gradient for all objects.
		G = sh.Gradients(min_weight, alpha, l2)
		// Update positions using gradient descent.
		sh.GradientDescent(step_size, G, 0.01)
		// Calculate current error.
		mE = 0.0
		for o = 0; o < sh.no; o++ {
			mE += G[o].error
		}
		mE /= float64(sh.no)
		// Calculate average distance from origin.
		mD = 0.0
		for o = 0; o < sh.no; o++ {
			mD += VectorMagnitude(sh.L[o])
		}
		mD /= float64(sh.no)
		// Report status.
		fmt.Printf("Epoch %6d: E=%.8e S=%.8e D=%.8e (alpha=%.4e; epoch took %v; %v elapsed).\n", epoch+1, mE, step_size, mD, alpha, time.Now().Sub(t), time.Now().Sub(T))
		// Write position of objects.
		if output_prefix != "" {
			sh.WriteLocations(fmt.Sprintf("%v_%v.csv", output_prefix, epoch+1))
		}
	}
}

func (sh *Shoehorn) LearnRprop(step_size float64, l2 float64, alpha float64, numepochs int, output_prefix string) {
	/*
		Uses the Rprop algorithm to find the best location for objects.
	*/
	var (
		epoch, o, j            int
		min_weight, mE, mS, mD float64
		S                      [][]float64
		G0, G1                 []GradientInfo
		T, t                   time.Time
	)
	// Initialization.
	T = time.Now()
	min_weight = 0.0
	S = ReturnMatrix(sh.no, sh.nd)
	for o = 0; o < sh.no; o++ {
		for j = 0; j < sh.nd; j++ {
			S[o][j] = step_size
		}
	}
	// Perform learning.
	for epoch = 0; epoch < numepochs; epoch++ {
		t = time.Now()
		// Get gradient for all objects.
		G1 = sh.Gradients(min_weight, alpha, l2)
		// Update positions using Rprop algorithm.
		S = sh.Rprop(S, G0, G1, 1e-3, step_size*10.)
		G0 = G1
		// Calculate current error.
		mE = 0.0
		for o = 0; o < len(G1); o++ {
			mE += G1[o].error
		}
		mE /= float64(sh.no)
		// Calculate current average step size.
		mS = 0.0
		for o = 0; o < len(S); o++ {
			for j = 0; j < len(S[o]); j++ {
				mS += S[o][j]
			}
		}
		mS /= float64(sh.no * sh.nd)
		// Calculate average distance from origin.
		mD = 0.0
		for o = 0; o < sh.no; o++ {
			mD += VectorMagnitude(sh.L[o])
		}
		mD /= float64(sh.no)
		// Report status.
		fmt.Printf("Epoch %6d: E=%.8e S=%.8e D=%.8e (alpha=%.4e; epoch took %v; %v elapsed).\n", epoch+1, mE, mS, mD, alpha, time.Now().Sub(t), time.Now().Sub(T))
		// Write position of objects.
		if output_prefix != "" {
			sh.WriteLocations(fmt.Sprintf("%v_%v.csv", output_prefix, epoch+1))
		}
	}
}

func (sh *Shoehorn) LearnLineSearch(step_size float64, l2 float64, alpha float64, numepochs int, output_prefix string) {
	/*
		Performs line search along each gradient direction to find the best location for objects.
	*/
	var (
		epoch, object, o, j, try, max_tries int
		min_weight, step, e0, e1, gmag      float64
		L                                   []float64
		G0, G1                              []GradientInfo
		T, t                                time.Time
	)
	min_weight = 0.0
	max_tries = 50
	L = make([]float64, sh.nd)
	// Iterate through epochs.
	T = time.Now()
	for epoch = 0; epoch < numepochs; epoch++ {
		// Perform line search for each object.
		for object = 0; object < sh.no; object++ {
			t = time.Now()
			// Get gradient and baseline error for object.
			G0 = sh.Gradients(min_weight, alpha, l2)
			e0 = 0.0
			for o = 0; o < sh.no; o++ {
				e0 += G0[o].error
			}
			gmag = VectorMagnitude(G0[object].gradient)
			// Store current object location.
			for j = 0; j < sh.nd; j++ {
				L[j] = sh.L[object][j]
			}
			// Perform line search.
			e1 = math.MaxFloat64
			step = step_size
			for try = 0; try < max_tries; try++ {
				// Relocate object.
				for j = 0; j < sh.nd; j++ {
					sh.L[object][j] = L[j] - ((step / gmag) * G0[object].gradient[j])
				}
				// Get line search error.
				G1 = sh.Gradients(min_weight, alpha, l2)
				e1 = 0.0
				for o = 0; o < sh.no; o++ {
					e1 += G1[o].error
				}
				// Terminate search if error improved, else reduce step size along line.
				if e1 < e0 {
					fmt.Printf("epoch=%v object=%v try=%v step=%.4e e0=%.6e e1=%.6e (took %v; %v elapsed).\n", epoch, object, try, step, e0, e1, time.Now().Sub(t), time.Now().Sub(T))
					break
				} else {
					step *= 0.5
				}
			}
			// Reset location if a better one wasn't found.
			if try == max_tries {
				for j = 0; j < sh.nd; j++ {
					sh.L[object][j] = L[j]
				}
			}
			// Write position of objects.
			if output_prefix != "" {
				sh.WriteLocations(fmt.Sprintf("%v_%v.csv", output_prefix, epoch+1))
			}
		}
	}
}

//
// Reconstruction methods.
//

func (sh *Shoehorn) Reconstructions(min_weight float64) (WP [][]float64, W []float64) {
	var (
		o       int
		channel chan int
	)
	// Initialization.
	runtime.GOMAXPROCS(runtime.NumCPU())
	channel = make(chan int, sh.no)
	WP = make([][]float64, sh.no)
	W = make([]float64, sh.no)
	// Create goroutines to compute reconstruction of each object.
	for o = 0; o < sh.no; o++ {
		go sh.ReconstructionWrapper(o, min_weight, WP, W, channel)
	}
	// Wait for all goroutines to signal completion.
	for o = 0; o < sh.no; o++ {
		<-channel
	}
	return
}

func (sh *Shoehorn) ReconstructionWrapper(object int, min_weight float64, WP [][]float64, W []float64, channel chan int) {
	WP[object], W[object] = sh.Reconstruction(object, min_weight)
	channel <- 0
	return
}

func (sh *Shoehorn) Reconstruction(object int, min_weight float64) (wp []float64, w float64) {
	var (
		f int
		n Neighbor
	)
	wp = make([]float64, sh.nf)
	for _, n = range sh.Neighbors(object, min_weight) {
		w += n.weight
		for f = 0; f < sh.nf; f++ {
			wp[f] += (n.weight * sh.O[n.object][f])
		}
	}
	return
}

//
// Gradient methods.
//

func (sh *Shoehorn) Gradients(min_weight float64, alpha float64, l2 float64) (G []GradientInfo) {
	var (
		o       int
		g       GradientInfo
		channel chan GradientInfo
	)
	runtime.GOMAXPROCS(runtime.NumCPU())
	// Precompute reconstruction data.
	WP, W := sh.Reconstructions(min_weight)
	// Compute gradient information.
	channel = make(chan GradientInfo, sh.no)
	for o = 0; o < sh.no; o++ {
		go sh.GradientWrapper(o, min_weight, alpha, l2, WP, W, channel)
	}
	// Sort gradient information by increasing object ID.
	G = make([]GradientInfo, sh.no)
	for o = 0; o < sh.no; o++ {
		g = <-channel
		G[g.object] = g
	}
	return
}

func (sh *Shoehorn) GradientWrapper(object int, min_weight float64, alpha float64, l2 float64, WP [][]float64, W []float64, channel chan GradientInfo) {
	g, e := sh.Gradient(object, min_weight, alpha, l2, WP, W)
	channel <- GradientInfo{object: object, gradient: g, error: e}
	return
}

func (sh *Shoehorn) Gradient(object int, min_weight float64, alpha float64, l2 float64, WP [][]float64, W []float64) (gradient []float64, error float64) {
	var (
		o, j, f                                  int
		distance, weight, p, q, tmp1, tmp2, tmp3 float64
		T1, T2                                   []float64
		N                                        Neighbors
		n                                        Neighbor
	)
	gradient = make([]float64, sh.nd)
	T1 = make([]float64, sh.nd)
	T2 = make([]float64, sh.nd)
	// Compute impact of object position on its own reconstruction error.
	N = sh.Neighbors(object, min_weight)
	for f = 0; f < sh.nf; f++ {
		p = sh.O[object][f]
		// Calculate the gradient terms.
		for j = 0; j < sh.nd; j++ {
			T1[j], T2[j] = 0.0, 0.0
		}
		for _, n = range N {
			tmp1 = n.weight / n.distance
			for j = 0; j < sh.nd; j++ {
				tmp2 = tmp1 * (sh.L[n.object][j] - sh.L[object][j])
				T1[j] += tmp2 * sh.O[n.object][f]
				T2[j] += tmp2
			}
		}
		// Set reconstruction probability.
		q = (alpha * p) + ((1.0 - alpha) * (WP[object][f] / W[object]))
		// Update gradient information.
		//tmp1 = (alpha - 1.0) * p / q
		tmp1 = 2.0 * (p - q) * (alpha - 1.0)
		for j = 0; j < sh.nd; j++ {
			gradient[j] += tmp1 * (((T1[j] * W[object]) - (WP[object][f] * T2[j])) / (W[object] * W[object]))
		}
		// Update error.
		//error += p * math.Log(p/q)
		error += math.Pow(p-q, 2.0)
	}
	// Compute impact of object position on reconstruction error of other objects.
	for o = 0; o < sh.no; o++ {
		if o != object {
			// Calculate distance and weight between current object and object being reconstructed.
			distance = 0.0
			for j = 0; j < sh.nd; j++ {
				distance += math.Pow(sh.L[object][j]-sh.L[o][j], 2.0)
			}
			distance = math.Pow(distance, 0.5)
			weight = math.Exp(-distance)
			tmp1 = weight / distance
			// Iterate over features of object getting reconstructed.
			for f = 0; f < sh.nf; f++ {
				p = sh.O[o][f]
				// Update gradient information.
				//tmp2 = (alpha - 1.0) * p / ((alpha * p) + ((1.0 - alpha) * (R.WPS[o][feature] / R.WS[o])))
				tmp2 = 2.0 * (p - ((alpha * p) + ((1.0 - alpha) * (WP[o][f] / W[o])))) * (alpha - 1.0)
				for j = 0; j < sh.nd; j++ {
					tmp3 = tmp1 * (sh.L[o][j] - sh.L[object][j])
					gradient[j] += tmp2 * (((W[o] * tmp3 * sh.O[object][f]) - (WP[o][f] * tmp3)) / (W[o] * W[o]))
				}
			}
		}
	}
	// Account for L2 punishment.
	for j = 0; j < sh.nd; j++ {
		error += l2 * math.Pow(sh.L[object][j], 2.0)
		gradient[j] += 2.0 * l2 * sh.L[object][j]
	}
	return
}

// Gradient descent methods.

func (sh *Shoehorn) GradientDescent(step_size float64, G []GradientInfo, downsample float64) {
	var (
		o, j  int
		scale float64
	)
	for o = 0; o < len(G); o++ {
		if rand.Float64() < downsample {
			scale = step_size / VectorMagnitude(G[o].gradient)
			for j = 0; j < sh.nd; j++ {
				sh.L[o][j] -= scale * G[o].gradient[j]
			}
		}
	}
	return
}

func (sh *Shoehorn) Rprop(S [][]float64, G0 []GradientInfo, G1 []GradientInfo, step_size_min float64, step_size_max float64) [][]float64 {
	var (
		o, j                       int
		gradient_product, inc, dec float64
	)
	inc = 1.01
	dec = 0.5
	for o = 0; o < len(G1); o++ {
		for j = 0; j < sh.nd; j++ {
			if G0 != nil {
				// Update the step size (consistent gradient directions get a boost, inconsistent directions get reduced).
				gradient_product = G0[o].gradient[j] * G1[o].gradient[j]
				if gradient_product > 0.0 {
					S[o][j] *= inc
				} else if gradient_product < 0.0 {
					S[o][j] *= dec
				}
				// Apply caps.
				if S[o][j] < step_size_min {
					S[o][j] = step_size_min
				}
				if S[o][j] > step_size_max {
					S[o][j] = step_size_max
				}
			}
			// Update the position based on the sign of its magnitude and the learned step size (RProp doesn't use gradient magnitudes).
			sh.L[o][j] -= math.Copysign(S[o][j], G1[o].gradient[j])
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
	for o = 0; o < sh.no; o++ {
		if o != object {
			// Calculate distance.
			distance = 0.0
			for j = 0; j < sh.nd; j++ {
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
	for id := 0; id < sh.no; id++ {
		object_ids = append(object_ids, id)
	}
	return
}

func (sh *Shoehorn) NormalizeObjects(metric float64) {
	/*
		Normalizes the magnitude of object representations.
	*/
	var (
		o, f int
		mag  float64
	)
	for o = 0; o < sh.no; o++ {
		mag = 0.0
		for f = 0; f < sh.nf; f++ {
			mag += math.Pow(sh.O[o][f], metric)
		}
		mag = math.Pow(mag, 1.0/metric)
		for f = 0; f < sh.nf; f++ {
			sh.O[o][f] /= mag
		}
	}
	return
}

// Returns information about the distance of points from the origin.
func (sh *Shoehorn) DistanceInformation() (min, mean, max float64) {
	min, mean, max = math.MaxFloat64, 0.0, 0.0
	for o := 0; o < sh.no; o++ {
		distance := VectorMagnitude(sh.L[o])
		if distance < min {
			min = distance
		}
		mean += distance
		if distance > max {
			max = distance
		}
	}
	mean /= float64(sh.no)
	return
}

func (sh *Shoehorn) Rescale(radius float64) {
	var (
		o, j int
	)
	// Calculate centroid and maximum distance from origin.
	centroid := make([]float64, sh.nd)
	max_distance := 0.0
	for o = 0; o < sh.no; o++ {
		for j = 0; j < sh.nd; j++ {
			centroid[j] += sh.L[o][j]
		}
		distance := VectorMagnitude(sh.L[o])
		if distance > max_distance {
			max_distance = distance
		}
	}
	for j = 0; j < sh.nd; j++ {
		centroid[j] /= float64(sh.no)
	}
	// Recenter and rescale each location.
	for o = 0; o < sh.no; o++ {
		for j = 0; j < sh.nd; j++ {
			sh.L[o][j] = (sh.L[o][j] - centroid[j]) * (radius / max_distance)
		}
	}
	return
}

func (sh *Shoehorn) WriteLocations(path string) {
	/*
		Writes the current locations of objects to a file.
	*/
	// Initialize the output file.
	of, err := os.Create(path)
	if err != nil {
		log.Fatal(err)
	}
	defer of.Close()
	// Write object locations to file.
	for object_name, object := range sh.oixs {
		line := make([]string, 0)
		line = append(line, object_name)
		for j := 0; j < sh.nd; j++ {
			line = append(line, fmt.Sprintf("%v", sh.L[object][j]))
		}
		of.Write([]byte(fmt.Sprintf("%v\n", strings.Join(line, ","))))
	}
}

func NewShoehorn(filename string, ndims int, downsample float64) (sh *Shoehorn) {
	/*
		Reads a CSV file of {object, feature, value} triples and parse them to create a shoehorn object.
	*/
	var (
		bfr                  *bufio.Reader
		seenobjs, sampleobjs map[string]bool
	)
	seenobjs = make(map[string]bool)
	sampleobjs = make(map[string]bool)
	// Seed the random number generator.
	rand.Seed(time.Now().Unix())
	// Open the file for reading.
	fh, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer fh.Close()
	// Read the lines of the file one at a time and grab them if sampled.
	S := make([][]string, 0)
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
			// If the object has not been seen before, decide whether to include it.
			_, seen := seenobjs[strvals[0]]
			if !seen {
				if rand.Float64() < downsample {
					sampleobjs[strvals[0]] = true
				}
			}
			seenobjs[strvals[0]] = true
			// Add the data to the store if the object is to be sampled.
			if sampleobjs[strvals[0]] {
				S = append(S, strvals)
			}
		}
		// Read from the file for the next iteration.
		line, isprefix, err = bfr.ReadLine()
	}
	// Create shoehorn object from selected data.
	sh = &Shoehorn{}
	sh.Create(S, ndims)
	return
}
