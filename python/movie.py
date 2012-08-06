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

def makeplot(locations_directory, filename, axis_limit=1.0):
	# Extract frame number.
	m = re.match("locations_(?P<num>\d+).csv", filename)
	frame = int(m.groups(0)[0])
	# load the data.
	data = loaddata(os.path.join(locations_directory, filename))
	# Plot the data.
	fig = lb.figure()
	for n in sorted(data):
		dat = np.array(data[n], "d")
		lb.plot(dat[:, 0], dat[:, 1], symbols[n], label='{}'.format(n))
	lb.axis([-axis_limit, axis_limit, -axis_limit, axis_limit])
	plotname = os.path.join(locations_directory, "{:06d}.png".format(frame))
	lb.savefig(plotname)
	lb.close(fig)

if __name__ == "__main__":
	# Extra locations directory.
	locations_directory = os.path.abspath(sys.argv[1])	
	# Get a list of all the files.
	fnames = [fname for fname in os.listdir(locations_directory) if re.match(r"locations_(?P<num>\d+)\.csv", fname)]
	# Find axis limit.
	axis_limit = None
	for fname in fnames:
		D = loaddata(os.path.join(locations_directory, fname))
		for n in D:
			dat = np.array(D[n], "d")
			extremum = np.abs(dat).max()
			if axis_limit is None or extremum > axis_limit:
				axis_limit = extremum
	axis_limit *= 1.1
	print("Axis limit is {:e}.".format(axis_limit))
	# Make the plots.
	for fname in fnames:
		makeplot(locations_directory, fname, axis_limit=axis_limit)
	# Make the movie.
	os.system("ffmpeg -y -i '{:s}' {:s}".format(os.path.join(locations_directory, "%06d.png"), os.path.join(locations_directory, "movie.m4v")))
	# Clean up the plots.
	os.system("rm {:s}".format(os.path.join(locations_directory, "*.png")))