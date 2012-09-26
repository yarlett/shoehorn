package shoehorn

import (
	"math"
)

//
// Definitions allow objects to be sorted in order of decreasing weight.
//

type Neighbor struct {
	object   int
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
	var step float64
	switch {
	case current_epoch < epoch0:
		parameter = val0
	case (current_epoch >= epoch0) && (current_epoch < epoch1):
		step = (val1 - val0) / float64(epoch1-epoch0)
		parameter = val0 + (float64(current_epoch-epoch0) * step)
	case current_epoch >= epoch1:
		parameter = val1
	}
	return
}

// Miscellaneous vector functions.

func VectorMagnitude(V []float64) (mag float64) {
	for i := 0; i < len(V); i++ {
		mag += math.Pow(V[i], 2.0)
	}
	mag = math.Pow(mag, 0.5)
	return
}

func ReturnMatrix(i, j int) (M [][]float64) {
	M = make([][]float64, i)
	for x := 0; x < i; x++ {
		M[x] = make([]float64, j)
	}
	return
}