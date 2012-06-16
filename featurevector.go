package shoehorn

type FeatureVector struct {
	data map[int]float64
}

func (fv *FeatureVector) Set(feature int, value float64) {
	fv.data[feature] = value
}

func (fv *FeatureVector) Sum() (sum float64) {
	for _, v := range fv.data {
		sum += v
	}
	return
}
