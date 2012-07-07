package shoehorn

import (
	"math"
)

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

func (fv *FeatureVector) KLDivergence(fvo *FeatureVector, alpha float64) (kl float64) {
	for feature, p := range fv.data {
		kl += (p * (math.Log(p) - math.Log((alpha*p)+((1.0-alpha)*fvo.data[feature]))))
	}
	return
}
