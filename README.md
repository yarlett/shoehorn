Shoehorn
========

Shoehorn is a nonlinear dimensional reduction tool written in Go. The shoehorn algorithm attempts to locate high-dimensional discrete probability distributions in a low-dimensional space with a specified number of dimensions. The algorithm operates by attempting to reconstruct the high-dimensional patterns by summing over the neighbors of these patterns in the low-dimensional space, where the weight attaching to a neighbor is a monotonically decreasing function of distance, so as to minimize the reconstruction error. Further technical details are provided in the [/report](https://github.com/yarlett/shoehorn/tree/master/report) directory.

The package implements 2 reconstruction methods, one involving exponential decay of summation weights, and one involving power-law decay of summation weights, along with gradient-descent learning algorithms for both. The code will automatically take advantage of multiple CPU cores in order to speed up learning.

The software takes input from a file consisting of {object_id, feature_id, feature_value} triples. Learning is performed according to a number of specifiable parameters and after learning has occurred the learned low-dimensional locations of the objects can be written in CSV format to a file. An example learning script can be found in the [/mnist](https://github.com/yarlett/shoehorn/tree/master/mnist) directory.