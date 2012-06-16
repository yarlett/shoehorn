package shoehorn

//
// Definitions allow objects to be sorted in order of decreasing weight.
//

type WeightPair struct {
	object_ix int
	d         float64
	distance  float64
	weight    float64
}

type Weights []WeightPair

func (w Weights) Len() int           { return len(w) }
func (w Weights) Swap(i, j int)      { w[i], w[j] = w[j], w[i] }
func (w Weights) Less(i, j int) bool { return w[i].weight > w[j].weight }

//
// GradientInfo allows gradient information about an object to be passed around in a channel.
//

type GradientInfo struct {
	object_ix int
	gradient  []float64
	error     float64
	lr        float64
}
