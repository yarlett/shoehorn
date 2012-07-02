package shoehorn

//
// Definitions allow objects to be sorted in order of decreasing weight.
//

type WeightPair struct {
	object   int
	distance float64
	weight   float64
}

type Weights []WeightPair

func (w Weights) Len() int           { return len(w) }
func (w Weights) Swap(i, j int)      { w[i], w[j] = w[j], w[i] }
func (w Weights) Less(i, j int) bool { return w[i].weight > w[j].weight }

//
// Gradient information that can be passed around in a channel.
//

type GradientInfo struct {
	object   int
	gradient []float64
}

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
