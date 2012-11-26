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
	// Neighbor information.
	ND [][]float64
	NW [][]float64
	// Reconstruction information.
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
	sh.ND = make([][]float64, sh.no)
	sh.NW = make([][]float64, sh.no)
	for o := 0; o < sh.no; o++ {
		sh.ND[o] = make([]float64, sh.no)
		sh.NW[o] = make([]float64, sh.no)
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

func (sh *Shoehorn) LearnGradientDescent(lr, mom, l2 float64, numepochs int, output_prefix string) {
	/*
		Uses gradient descent to find the best location for objects.
	*/
	var (
		epoch int
		E     float64
		U     [][]float64
		T, t  time.Time
	)
	U = ReturnMatrix(sh.no, sh.nd, 0)
	// Initialization.
	T = time.Now()
	// Perform learning.
	for epoch = 0; epoch < numepochs; epoch++ {
		t = time.Now()
		// Get gradient for all objects.
		sh.SetGradients(l2)
		// Calculate current error.
		E = sh.CurrentError()
		// Update positions using gradient descent.
		sh.GradientDescent(lr, mom, U)
		// Report status.
		fmt.Printf("Epoch %6d: E=%.8e G=%.8e S=%.8e D=%.8e (epoch took %v; %v elapsed).\n", epoch+1, E, MeanMagnitude(sh.G), lr, MeanMagnitude(sh.L), time.Now().Sub(t), time.Now().Sub(T))
		// Write position of objects.
		if output_prefix != "" {
			sh.WriteLocations(fmt.Sprintf("%v.csv", output_prefix))
		}
	}
}

func (sh *Shoehorn) LearnRprop(initial_step_size float64, l2 float64, numepochs int, output_prefix string) {
	/*
		Uses gradient descent to find the best location for objects.
	*/
	var (
		epoch int
		E     float64
		S, G  [][]float64
		T, t  time.Time
	)
	// Initialization.
	T = time.Now()
	S = ReturnMatrix(sh.no, sh.nd, initial_step_size)
	G = ReturnMatrix(sh.no, sh.nd, 0.)
	// Perform learning.
	for epoch = 0; epoch < numepochs; epoch++ {
		t = time.Now()
		// Get gradient for all objects.
		sh.SetGradients(l2)
		// Calculate current error.
		E = sh.CurrentError()
		// Update positions using gradient descent.
		S = sh.Rprop(S, G, sh.G, 1e-6, 1e-1)
		for o := 0; o < sh.no; o++ {
			for j := 0; j < sh.nd; j++ {
				G[o][j] = sh.G[o][j]
			}
		}
		// Report status.
		fmt.Printf("Epoch %6d: E=%.8e G=%.8e S=%.8e D=%.8e (epoch took %v; %v elapsed).\n", epoch+1, E, MeanMagnitude(sh.G), MeanMagnitude(S), MeanMagnitude(sh.L), time.Now().Sub(t), time.Now().Sub(T))
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
		gmag, E0, E1, step                 float64
		L                                  []float64
		T, t1, t2                          time.Time
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
			sh.SetGradients(l2)
			E0 = sh.CurrentError()
			gmag = VectorMagnitude(sh.G[object])
			// Save current location of object.
			copy(L, sh.L[object])
			// Set the initial step size to the furthest object away.
			step = 0.0
			for o = 0; o < len(sh.ND[object]); o++ {
				if sh.ND[object][o] > step {
					step = sh.ND[object][o]
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
				sh.SetErrors(l2)
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
		o1, o2, d int
	)
	// Set neighbor information (capitalizing on symmetry of distances and weights).
	for o1 = 0; o1 < sh.no; o1++ {
		for o2 = 0; o2 <= o1; o2++ {
			// Calculate distance and weight.
			sh.ND[o1][o2] = 0.
			for d = 0; d < sh.nd; d++ {
				sh.ND[o1][o2] += math.Pow(sh.L[o1][d]-sh.L[o2][d], 2.)
			}
			sh.ND[o1][o2] = math.Sqrt(sh.ND[o1][o2])
			sh.NW[o1][o2] = math.Exp(-sh.ND[o1][o2])
			// Set symmetric values.
			sh.ND[o2][o1] = sh.ND[o1][o2]
			sh.NW[o2][o1] = sh.NW[o1][o2]
		}
	}
	return
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
		sh.WP[object][f] = 0.
	}
	sh.W[object] = 0.
	// Compute the reconstruction information.
	for o = 0; o < sh.no; o++ {
		if o != object {
			for f = 0; f < sh.nf; f++ {
				sh.WP[object][f] += sh.NW[object][o] * sh.O[o][f]
			}
			sh.W[object] += sh.NW[object][o]
		}
	}
	// Signal completion.
	channel <- true
}

//
// Error methods.
//

func (sh *Shoehorn) SetErrors(l2 float64) {
	var (
		o       int
		channel chan bool
	)
	runtime.GOMAXPROCS(runtime.NumCPU())
	// Precompute reconstruction data.
	sh.SetNeighbors()
	sh.SetReconstructions()
	// Compute error for each object.
	channel = make(chan bool, sh.no)
	for o = 0; o < sh.no; o++ {
		go sh.Error(o, l2, channel)
	}
	// Retrieve errors.
	for o = 0; o < sh.no; o++ {
		<-channel
	}
	return
}

func (sh *Shoehorn) Error(object int, l2 float64, channel chan bool) {
	var (
		f int
	)
	// Initialize error.
	sh.E[object] = 0.
	// Compute reconstruction error for object.
	for f = 0; f < sh.nf; f++ {
		sh.E[object] += math.Pow(sh.O[object][f]-(sh.WP[object][f] / sh.W[object]), 2.)
	}
	// Account for L2 punishment.
	distance := VectorMagnitude(sh.L[object])
	sh.E[object] += l2 * distance
	// Signal completion.
	channel <- true
}

func (sh *Shoehorn) CurrentError() (E float64) {
	for o := 0; o < len(sh.E); o++ {
		E += sh.E[o]
	}
	E /= float64(sh.no)
	return
}

func (sh *Shoehorn) TotalEnergy(l2 float64) (E float64) {
	sh.SetErrors(l2)
	E = sh.CurrentError()
	return
}

//
// Gradient methods.
//

func (sh *Shoehorn) SetGradients(l2 float64) {
	var (
		o       int
		channel chan bool
	)
	runtime.GOMAXPROCS(runtime.NumCPU())
	// Precompute reconstruction data.
	sh.SetNeighbors()
	sh.SetReconstructions()
	// Compute gradient information.
	channel = make(chan bool, sh.no)
	for o = 0; o < sh.no; o++ {
		go sh.Gradient(o, l2, channel)
	}
	// Retrieve gradient information.
	for o = 0; o < sh.no; o++ {
		<-channel
	}
	return
}

func (sh *Shoehorn) Gradient(object int, l2 float64, channel chan bool) {
	var (
		o, d, f                                                           int
		E, g, h, g_other, h_other, gprime_other, hprime_other, tmp1, tmp2 float64
		gprime, hprime, G                                                 []float64
	)
	G = make([]float64, sh.nd)
	gprime = make([]float64, sh.nd)
	hprime = make([]float64, sh.nd)
	// Compute gradient information for each feature.
	for f = 0; f < sh.nf; f++ {
		// Reset gradient terms (they are computed on a per-feature basis).
		for d = 0; d < sh.nd; d++ {
			gprime[d], hprime[d] = 0., 0.
		}
		g = sh.WP[object][f]
		h = sh.W[object]
		for o = 0; o < sh.no; o++ {
			if o != object {
				// tmp1 = sh.NW[object][o] / sh.ND[object][o]
				tmp1 = 1. / ((1. + sh.ND[object][o]) * (1. + sh.ND[object][o]) * sh.ND[object][o])
				g_other = sh.WP[o][f]
				h_other = sh.W[o]
				for d = 0; d < sh.nd; d++ {
					// Accumulate information about the impact of object's location on its own reconstruction error.
					tmp2 = tmp1 * (sh.L[o][d] - sh.L[object][d])
					gprime[d] += sh.O[o][f] * tmp2
					hprime[d] += tmp2
					// Update gradient information about the impact of object's location on the reconstruction error of object o.
					gprime_other = sh.O[object][f] * tmp2
					hprime_other = tmp2
					G[d] += 2. * ((g_other / h_other) - sh.O[o][f]) * ((gprime_other * h_other) - (g_other * hprime_other)) / (h_other * h_other)
				}
			}
		}
		// Update gradient.
		tmp1 = 2. * ((g / h) - sh.O[object][f]) / (h * h)
		for d = 0; d < sh.nd; d++ {
			G[d] += tmp1 * ((gprime[d] * h) - (g * hprime[d]))
		}
		// Update error.
		E += math.Pow(sh.O[object][f]-(g/h), 2.)
	}
	// Account for L2 punishment.
	distance := VectorMagnitude(sh.L[object])
	E += l2 * distance
	for d = 0; d < sh.nd; d++ {
		G[d] += l2 * sh.L[object][d] / distance
	}
	// Set the error and gradient on the shoehorn object.
	sh.E[object] = E
	for d = 0; d < sh.nd; d++ {
		sh.G[object][d] = G[d]
	}
	// Signal completion.
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

func (sh *Shoehorn) GradientDescent(learning_rate, momentum float64, U [][]float64) {
	var (
		o, d   int
		u, mag, maxmag float64
	)
	// Scale gradients so largest has magnitude of 1.
	maxmag = 0.
	for o = 0; o < sh.no; o++ {
		mag = VectorMagnitude(sh.G[o])
		if mag > maxmag {
			maxmag = mag
		}
	}
	// Update locations.
	for o = 0; o < sh.no; o++ {
		// mag = VectorMagnitude(sh.G[o])
		for d = 0; d < sh.nd; d++ {
			u = (learning_rate * -sh.G[o][d] / maxmag) + (momentum * U[o][d])
			sh.L[o][d] += u
			U[o][d] = u
		}
	}
	return
}

func (sh *Shoehorn) Rprop(S [][]float64, G0 [][]float64, G1 [][]float64, step_size_min float64, step_size_max float64) [][]float64 {
	var (
		o, d  int
		gprod float64
	)
	for o = 0; o < len(G1); o++ {
		for d = 0; d < len(G1[o]); d++ {
			// Update the step size (consistent gradient directions get a boost, inconsistent directions get reduced).
			gprod = G0[o][d] * G1[o][d]
			if gprod > 0. {
				S[o][d] *= 1.1
			} else if gprod < 0. {
				S[o][d] *= 0.5
				G1[o][d] = 0.
			}
			// Apply caps.
			if S[o][d] < step_size_min {
				S[o][d] = step_size_min
			}
			if S[o][d] > step_size_max {
				S[o][d] = step_size_max
			}
			// Update the position based on the sign of its magnitude and the learned step size (RProp doesn't use gradient magnitudes).
			sh.L[o][d] -= math.Copysign(S[o][d], G1[o][d])
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
		o, d int
	)
	// Calculate centroid and maximum distance from origin.
	centroid := make([]float64, sh.nd)
	max_distance := 0.0
	for o = 0; o < sh.no; o++ {
		for d = 0; d < sh.nd; d++ {
			centroid[d] += sh.L[o][d]
		}
		distance := VectorMagnitude(sh.L[o])
		if distance > max_distance {
			max_distance = distance
		}
	}
	for d = 0; d < sh.nd; d++ {
		centroid[d] /= float64(sh.no)
	}
	// Recenter and rescale each location.
	for o = 0; o < sh.no; o++ {
		for d = 0; d < sh.nd; d++ {
			sh.L[o][d] = (sh.L[o][d] - centroid[d]) * (radius / max_distance)
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
	// rand.Seed(time.Now().Unix())
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
