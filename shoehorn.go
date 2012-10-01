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
	nd int
	no int
	nf int
	// Object information.
	oixs map[string]int
	fixs map[string]int
	O    [][]float64
	L    [][]float64
	// Reconstruction information.
	N  [][]Neighbor
	WP [][]float64
	W  []float64
	// Error and gradient information.
	E []float64
	G [][]float64
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
	// Create storage for nearest neighbors.
	sh.N = make([][]Neighbor, sh.no)
	for o := 0; o < sh.no; o++ {
		sh.N[o] = make([]Neighbor, sh.no)
	}
	// Create objects and storage for reconstructions.
	sh.O = make([][]float64, sh.no)
	sh.WP = make([][]float64, sh.no)
	for o := 0; o < sh.no; o++ {
		sh.O[o] = make([]float64, sh.nf)
		sh.WP[o] = make([]float64, sh.nf)
	}
	sh.W = make([]float64, sh.no)
	// Create storage for rror and gradient information.
	sh.E = make([]float64, sh.no)
	sh.G = make([][]float64, sh.no)
	for o := 0; o < sh.no; o++ {
		sh.G[o] = make([]float64, sh.nd)
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

func (sh *Shoehorn) LearnGradientDescent(step_size float64, alpha float64, l2 float64, numepochs int, output_prefix string) {
	/*
		Uses gradient descent to find the best location for objects.
	*/
	var (
		epoch, o int
		e, mD    float64
		T, t     time.Time
	)
	// Initialization.
	T = time.Now()
	// Perform learning.
	for epoch = 0; epoch < numepochs; epoch++ {
		t = time.Now()
		// Get gradient for all objects.
		sh.SetGradients(alpha, l2)
		// Calculate current error.
		e = sh.CurrentError()
		// Update positions using gradient descent.
		noise := ParameterSetter(epoch, 0, 1.0, numepochs, 0.0)
		sh.GradientDescent(step_size, noise)
		// Calculate average distance from origin.
		mD = 0.0
		for o = 0; o < sh.no; o++ {
			mD += VectorMagnitude(sh.L[o])
		}
		mD /= float64(sh.no)
		// Report status.
		fmt.Printf("Epoch %6d: E=%.8e S=%.8e D=%.8e (epoch took %v; %v elapsed).\n", epoch+1, e, step_size, mD, time.Now().Sub(t), time.Now().Sub(T))
		// Write position of objects.
		if output_prefix != "" {
			sh.WriteLocations(fmt.Sprintf("%v_%v.csv", output_prefix, epoch+1))
		}
	}
}

func (sh *Shoehorn) LearnLineSearch(alpha float64, l2 float64, numepochs int, output_prefix string) {
	/*
		Performs line search along each gradient direction to find the best location for objects.
	*/
	var (
		epoch, object, o, j, try, numtries int
		gmag, E0, E1, step              float64
		L                               []float64
		T, t1, t2                            time.Time
	)
	// Initialization.
	T = time.Now()
	numtries = 20
	L = make([]float64, sh.nd)
	// Perform epochs of learning.
	for epoch = 0; epoch < numepochs; epoch++ {
		// Cycle through objects in a random order.
		for _, object = range rand.Perm(sh.no) {
			t1 = time.Now()
			// Get gradients and baseline error.
			sh.SetGradients(alpha, l2)
			E0 = sh.CurrentError()
			gmag = VectorMagnitude(sh.G[object])
			// Save current location of object.
			copy(L, sh.L[object])
			// Set the initial step size to the furthest object away.
			step = 0.0
			for o = 0; o < len(sh.N[object]); o++ {
				if sh.N[object][o].distance > step {
					step = sh.N[object][o].distance
				}
			}
			// Perform line search.
			t2 = time.Now()
			for try = 0; try < numtries; try++ {
				// Relocate object.
				for j = 0; j < sh.nd; j++ {
					sh.L[object][j] = L[j] - ((step / gmag) * sh.G[object][j])
				}
				// Get error at this location.
				sh.SetErrors(alpha, l2)
				E1 = sh.CurrentError()
				// Terminate search if error reduced.
				if E1 < E0 {
					break
				} else {
					step *= 0.5
				}
			}
			// Reset object to original position if error not improved.
			if E1 >= E0 {
				copy(sh.L[object], L)
				E1 = E0
			}
			// Report status.
			fmt.Printf("Epoch %6d Object %6d: tries=%5d E0=%.8e E1=%.8e S=%.8e (alpha=%.4e; took %v took %v; %v elapsed).\n", epoch+1, object, try+1, E0, E1, step, alpha, time.Now().Sub(t1), time.Now().Sub(t2), time.Now().Sub(T))
			// Write position of objects.
			if output_prefix != "" {
				sh.WriteLocations(fmt.Sprintf("%v_%v.csv", output_prefix, epoch+1))
			}
		}
	}
}

//
// Neighbor information.
//

func (sh *Shoehorn) SetNeighbors() {
	var (
		object  int
		channel chan bool
	)
	// Initialization.
	runtime.GOMAXPROCS(runtime.NumCPU())
	channel = make(chan bool, sh.no)
	// Create goroutines to compute neighbor information for each object.
	for object = 0; object < sh.no; object++ {
		go sh.SetNeighbor(object, channel)
	}
	// Wait for all goroutines to signal completion.
	for object = 0; object < sh.no; object++ {
		<-channel
	}
	return
}

func (sh *Shoehorn) SetNeighbor(object int, channel chan bool) {
	/*
		Sets the neighbor information (distances and weights) for a specified object.
	*/
	var (
		o, j int
	)
	for o = 0; o < sh.no; o++ {
		if o != object {
			// Calculate distance and weight.
			sh.N[object][o].distance = 0.0
			for j = 0; j < sh.nd; j++ {
				sh.N[object][o].distance += math.Pow(sh.L[object][j]-sh.L[o][j], 2.0)
			}
			sh.N[object][o].distance = math.Pow(sh.N[object][o].distance, 0.5)
			sh.N[object][o].weight = math.Exp(-sh.N[object][o].distance)
		}
	}
	channel <- true
}

//
// Reconstruction methods.
//

func (sh *Shoehorn) SetReconstructions() {
	var (
		object  int
		channel chan bool
	)
	// Initialization.
	runtime.GOMAXPROCS(runtime.NumCPU())
	channel = make(chan bool, sh.no)
	// Set nearest neighbor information.
	sh.SetNeighbors()
	// Create goroutines to compute reconstruction of each object.
	for object = 0; object < sh.no; object++ {
		go sh.SetReconstruction(object, channel)
	}
	// Wait for all goroutines to signal completion.
	for object = 0; object < sh.no; object++ {
		<-channel
	}
	return
}

func (sh *Shoehorn) SetReconstruction(object int, channel chan bool) {
	/*
		Sets the nearest neighbor information and reconstruction information for the specified object.
	*/
	var (
		o, f int
	)
	// Reset the reconstruction information.
	for f = 0; f < sh.nf; f++ {
		sh.WP[object][f] = 0.0
	}
	sh.W[object] = 0.0
	// Compute the reconstruction information.
	for o = 0; o < sh.no; o++ {
		if o != object {
			for f = 0; f < sh.nf; f++ {
				sh.WP[object][f] += sh.N[object][o].weight * sh.O[o][f]
			}
			sh.W[object] += sh.N[object][o].weight
		}
	}
	channel <- true
}

//
// Error methods.
//

func (sh *Shoehorn) SetErrors(alpha float64, l2 float64) {
	var (
		o       int
		channel chan bool
	)
	runtime.GOMAXPROCS(runtime.NumCPU())
	// Precompute reconstruction data.
	sh.SetReconstructions()
	// Compute error for each object.
	channel = make(chan bool, sh.no)
	for o = 0; o < sh.no; o++ {
		go sh.Error(o, alpha, l2, channel)
	}
	// Retrieve errors.
	for o = 0; o < sh.no; o++ {
		<-channel
	}
	return
}

func (sh *Shoehorn) Error(object int, alpha float64, l2 float64, channel chan bool) {
	var (
		f, j int
		p, q float64
	)
	// Initialize error.
	sh.E[object] = 0.0
	// Compute reconstruction error for object.
	for f = 0; f < sh.nf; f++ {
		// Get actual and reconstructed feature values.
		p = sh.O[object][f]
		q = (alpha * p) + ((1.0 - alpha) * (sh.WP[object][f] / sh.W[object]))
		// Update error.
		//error += p * math.Log(p/q)
		sh.E[object] += math.Pow(p-q, 2.0)
	}
	// Account for L2 punishment.
	for j = 0; j < sh.nd; j++ {
		sh.E[object] += l2 * math.Pow(sh.L[object][j], 2.0)
	}
	channel <- true
}

func (sh *Shoehorn) CurrentError() (E float64) {
	for o := 0; o < len(sh.E); o++ {
		E += sh.E[o]
	}
	E /= float64(sh.no)
	return
}

//
// Gradient methods.
//

func (sh *Shoehorn) SetGradients(alpha float64, l2 float64) {
	var (
		o       int
		channel chan bool
	)
	runtime.GOMAXPROCS(runtime.NumCPU())
	// Precompute reconstruction data.
	sh.SetReconstructions()
	// Compute gradient information.
	channel = make(chan bool, sh.no)
	for o = 0; o < sh.no; o++ {
		go sh.Gradient(o, alpha, l2, channel)
	}
	// Retrieve gradient information.
	for o = 0; o < sh.no; o++ {
		<-channel
	}
	return
}

func (sh *Shoehorn) Gradient(object int, alpha float64, l2 float64, channel chan bool) {
	var (
		o, j, f                int
		p, q, tmp1, tmp2, tmp3 float64
		T1, T2                 []float64
		n                      Neighbor
	)
	T1 = make([]float64, sh.nd)
	T2 = make([]float64, sh.nd)
	for j = 0; j < sh.nd; j++ {
		sh.G[object][j] = 0.0
	}
	sh.E[object] = 0.0
	// Compute impact of object position on its own reconstruction error.
	for f = 0; f < sh.nf; f++ {
		p = sh.O[object][f]
		// Calculate the gradient terms.
		for j = 0; j < sh.nd; j++ {
			T1[j], T2[j] = 0.0, 0.0
		}
		for o, n = range sh.N[object] {
			if o != object {
				tmp1 = n.weight / n.distance
				for j = 0; j < sh.nd; j++ {
					tmp2 = tmp1 * (sh.L[o][j] - sh.L[object][j])
					T1[j] += tmp2 * sh.O[o][f]
					T2[j] += tmp2
				}
			}
		}
		// Set reconstruction probability.
		q = (alpha * p) + ((1.0 - alpha) * (sh.WP[object][f] / sh.W[object]))
		// Update gradient information.
		//tmp1 = (alpha - 1.0) * p / q
		tmp1 = 2.0 * (p - q) * (alpha - 1.0)
		for j = 0; j < sh.nd; j++ {
			sh.G[object][j] += tmp1 * (((T1[j] * sh.W[object]) - (sh.WP[object][f] * T2[j])) / (sh.W[object] * sh.W[object]))
		}
		// Update error.
		//sh.E[object] += p * math.Log(p/q)
		sh.E[object] += math.Pow(p-q, 2.0)
	}
	// Compute impact of object position on reconstruction error of other objects.
	for o = 0; o < sh.no; o++ {
		if o != object {
			// Calculate distance and weight between current object and object being reconstructed.
			tmp1 = sh.N[object][o].weight / sh.N[object][o].distance
			// Iterate over features of object getting reconstructed.
			for f = 0; f < sh.nf; f++ {
				p = sh.O[o][f]
				// Update gradient information.
				//tmp2 = (alpha - 1.0) * p / ((alpha * p) + ((1.0 - alpha) * (R.WPS[o][feature] / R.WS[o])))
				tmp2 = 2.0 * (p - ((alpha * p) + ((1.0 - alpha) * (sh.WP[o][f] / sh.W[o])))) * (alpha - 1.0)
				for j = 0; j < sh.nd; j++ {
					tmp3 = tmp1 * (sh.L[o][j] - sh.L[object][j])
					sh.G[object][j] += tmp2 * (((sh.W[o] * tmp3 * sh.O[object][f]) - (sh.WP[o][f] * tmp3)) / (sh.W[o] * sh.W[o]))
				}
			}
		}
	}
	// Account for L2 punishment.
	for j = 0; j < sh.nd; j++ {
		sh.E[object] += l2 * math.Pow(sh.L[object][j], 2.0)
		sh.G[object][j] += 2.0 * l2 * sh.L[object][j]
	}
	channel <- true
}

func (sh *Shoehorn) CopyGradient() (G [][]float64) {
	G = make([][]float64, sh.no)
	for o := 0; o < sh.no; o++ {
		G[o] = make([]float64, sh.nd)
		for j := 0; j < sh.nd; j++ {
			G[o][j] = sh.G[o][j]
		}
	}
	return
}

// Gradient descent methods.

func (sh *Shoehorn) GradientDescent(step_size float64, noise float64) {
	var (
		o, j  int
		scale float64
	)
	for o = 0; o < len(sh.G); o++ {
		scale = step_size / VectorMagnitude(sh.G[o])
		for j = 0; j < sh.nd; j++ {
			sh.L[o][j] -= scale * sh.G[o][j]
			sh.L[o][j] += (rand.NormFloat64() * noise)
		}
	}
	return
}

func (sh *Shoehorn) Rprop(S [][]float64, G0 []GradientInfo, G1 []GradientInfo, step_size_min float64, step_size_max float64, downsample float64) [][]float64 {
	var (
		o, j                       int
		gradient_product, inc, dec float64
		move                       bool
	)
	inc = 1.01
	dec = 0.5
	for o = 0; o < len(G1); o++ {
		if rand.Float64() < downsample {
			move = true
		} else {
			move = false
		}
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
			if move {
				sh.L[o][j] -= math.Copysign(S[o][j], G1[o].gradient[j])
			}
		}
	}
	return S
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

func (sh *Shoehorn) LimitLocations(radius float64) {
	for o := 0; o < sh.no; o++ {
		mg := VectorMagnitude(sh.L[o])
		if mg > radius {
			scale := radius / mg
			for j := 0; j < sh.nd; j++ {
				sh.L[o][j] *= scale
			}
		}
	}
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
