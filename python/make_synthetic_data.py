import csv, numpy as np, random

def distort(prototype, noise=0.50):
	multipliers = 1.0 + ((2.0 * (np.random.random(prototype.size) - 0.5)) * noise)
	distortion = prototype * multipliers
	distortion /= distortion.sum()
	return distortion

if __name__ == "__main__":
	"""
	Generates some probability distributions centered around a specified number of object types.
	"""

	# Parameters.
	num_types = 5
	num_objects = 200
	num_features = 10
	noise = 0.5

	# Generate item prototypes.
	prototypes = []
	for _ in xrange(num_types):
		prototypes.append(np.random.random(num_features))

	# Generate data.
	data = []
	for _ in xrange(num_objects):
		object_type = random.randint(0, num_types-1)
		data.append((object_type, distort(prototypes[object_type], noise=noise)))

	# Write data to CSV.
	N = {}
	fh = open("/Users/yarlett/mygo/src/shoehorn/data/synthetic_data.csv", "w")
	writer = csv.writer(fh)
	for object_type, object_features in data:
		# Get object number.
		object_id = N.get(object_type, 0)
		N[object_type] = object_id + 1
		# Write feature data.
		for feature_id in xrange(object_features.size):
			writer.writerow(("{:d}_{:d}".format(object_type, object_id), feature_id, object_features[feature_id]))
	fh.close()