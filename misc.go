package shoehorn

import (
	"math"
	"sort"
)

//
// Definitions allow objects to be sorted in order of decreasing weight.
//

type Neighbor struct {
	distance float64
	weight   float64
}

type Neighbors []Neighbor

func (N Neighbors) Len() int           { return len(N) }
func (N Neighbors) Swap(i, j int)      { N[i], N[j] = N[j], N[i] }
func (N Neighbors) Less(i, j int) bool { return N[i].weight > N[j].weight }

//
// Gradient information that can be passed around in a channel.
//

type GradientInfo struct {
	object   int
	gradient []float64
	error    float64
}

type GradientInfos []GradientInfo

func (GIS GradientInfos) Len() int           { return len(GIS) }
func (GIS GradientInfos) Swap(i, j int)      { GIS[i], GIS[j] = GIS[j], GIS[i] }
func (GIS GradientInfos) Less(i, j int) bool { return GIS[i].object < GIS[j].object }

//
// Function that returns the required value of a parameter given the current epoch and the start and end points of the parameter.
//

func ParameterSetter(current_epoch int, epoch0 int, val0 float64, epoch1 int, val1 float64) (parameter float64) {
	//var step float64
	switch {
	case current_epoch < epoch0:
		parameter = val0
	case (current_epoch >= epoch0) && (current_epoch < epoch1):
		// step = (val1 - val0) / float64(epoch1-epoch0)
		// parameter = val0 + (float64(current_epoch-epoch0) * step)
		multiplier := math.Exp(math.Log(val1/val0) / float64(epoch1-epoch0))
		parameter = val0 * math.Pow(multiplier, float64(current_epoch-epoch0))
	case current_epoch >= epoch1:
		parameter = val1
	}
	return
}

// Scheduling functions (e.g. for temperatures in simulated annealing).

func GetLinearSchedule(value0, value1 float64, num_values int) (values []float64) {
	var i int
	var alpha float64 = (value0 - value1) / float64(num_values-1)
	for i = 0; i < num_values; i++ {
		values = append(values, value0-float64(i)*alpha)
	}
	return
}

func GetExponentialSchedule(value0, value1 float64, num_values int) (values []float64) {
	var i int
	var alpha float64
	alpha = math.Pow(value1/value0, 1./float64(num_values-1))
	for i = 0; i < num_values; i++ {
		values = append(values, value0*math.Pow(alpha, float64(i)))
	}
	return
}

// Quantile function. Returns quantile sample of values for a given percentile.

func Quantile(data []float64, percentile float64) (quantile float64) {
	// Ensure the data is sorted.
	sort.Float64s(data)
	// Ensure the percentile is within acceptable range.
	if percentile < 0. {
		percentile = 0.
	}
	if percentile > 1. {
		percentile = 1.
	}
	// Get the integer and remainder parts of the index.
	ix_all := percentile * float64(len(data)-1)
	ix_int := int(math.Floor(ix_all))
	ix_rem := ix_all - float64(ix_int)
	// Set quantile, applying linear interpolation if possible.
	if ix_int <= 0 {
		quantile = data[0]
	} else if ix_int >= len(data)-1 {
		quantile = data[len(data)-1]
	} else {
		quantile = data[ix_int] + (ix_rem * (data[ix_int-1] - data[ix_int]))
	}
	return
}

// Miscellaneous vector functions.

func VectorMagnitude(V []float64) (mag float64) {
	for i := 0; i < len(V); i++ {
		mag += math.Pow(V[i], 2.)
	}
	mag = math.Sqrt(mag)
	return
}

func MeanMagnitude(M [][]float64) (magnitude float64) {
	for i := 0; i < len(M); i++ {
		magnitude += VectorMagnitude(M[i])
	}
	magnitude /= float64(len(M))
	return
}

func ReturnMatrix(i, j int, initialval float64) (M [][]float64) {
	M = make([][]float64, i)
	for ii := 0; ii < i; ii++ {
		M[ii] = make([]float64, j)
		for jj := 0; jj < j; jj++ {
			M[ii][jj] = initialval
		}
	}
	return
}
