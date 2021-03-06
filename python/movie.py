import csv, numpy as np, os, pylab as lb, re, sys

symbols = {
	"0": ".", "1": "o", "2": "v", "3": "^", "4": "+", "5": "x", "6": "d", "7": "*", "8": "s", "9": "h",
}

def loaddata(filename):
	# Read in the data.
	f = open(filename, "r")
	reader = csv.reader(f)
	D = {}
	for row in reader:
		obj_name = row[0][0]
		if obj_name not in D:
			D[obj_name] = []
		D[obj_name].append([float(v) for v in row[1:]])
	f.close()
	return D

def makeplot(locations_directory, filename, prefix, axis_limit=None):
	# Extract frame number.
	m = re.match("{:s}.*?(?P<num>\d+).csv".format(prefix), filename)
	frame = int(m.groups(0)[0])
	# Load the data.
	data = loaddata(os.path.join(locations_directory, filename))
	# Plot the data.
	fig = lb.figure()
	for n in sorted(data):
		dat = np.array(data[n], "d")
		lb.plot(dat[:, 0], dat[:, 1], symbols[n], label='{}'.format(n))
	if axis_limit is not None:
		lb.axis([-axis_limit, axis_limit, -axis_limit, axis_limit])
	plotname = os.path.join(locations_directory, "{:06d}.png".format(frame))
	lb.savefig(plotname)
	lb.close(fig)

if __name__ == "__main__":
	# Find the most recently modified CSV file matching the prefix in sys.argv[1].
	directory, prefix = os.path.split(sys.argv[1])
	directory = os.path.abspath(directory)
	# Get a list of all the files.
	fnames = [fname for fname in os.listdir(directory) if re.match("{:s}.+\.csv".format(prefix), fname)]
	# # Find axis limit.
	# axis_limit = None
	# for fname in fnames:
	# 	D = loaddata(os.path.join(directory, fname))
	# 	for n in D:
	# 		dat = np.array(D[n], "d")
	# 		extremum = np.abs(dat).max()
	# 		if axis_limit is None or extremum > axis_limit:
	# 			axis_limit = extremum
	# axis_limit *= 1.1
	# print("Axis limit is {:e}.".format(axis_limit))
	axis_limit = None
	# Make the plots.
	for fname in fnames:
		makeplot(directory, fname, prefix, axis_limit=axis_limit)
	# Calculate frames per second to make movie last 1 minute.
	fps = 60.0 / float(len(fnames))
	# Make the movie.
	os.system("ffmpeg -y -i '{:s}' {:s}".format(os.path.join(directory, "%06d.png"), os.path.join(directory, "movie.m4v")))
	# Clean up the plots.
	os.system("rm {:s}".format(os.path.join(directory, "*.png")))