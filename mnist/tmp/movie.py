import csv, numpy as np, os, pylab as lb, re, sys
#import matplotlib.pyplot as plt

symbols = {
	'0': '.',
	'1': 'o',
	'2': 'v',
	'3': '^',
	'4': '+',
	'5': 'x',
	'6': 'd',
	'7': '*',
	'8': 's',
	'9': 'h',
}

def makeplot(filename, axis_limit=4.0):
	# Extract frame number.
	m = re.match("positions_(?P<num>\d+).csv", filename)
	frame = int(m.groups(0)[0])
	# Read in the data.
	f = open(filename, "r")
	reader = csv.reader(f)
	D = {}
	for row in reader:
		obj_name = row[0][0]
		if obj_name not in D:
			D[obj_name] = []
		D[obj_name].append([float(v) for v in row[1:]])
	# Plot the data.
	fig = lb.figure()
	for n in sorted(D):
		dat = np.array(D[n], "d")
		lb.plot(dat[:, 0], dat[:, 1], symbols[n], label='{}'.format(n))
	lb.axis([-axis_limit, axis_limit, -axis_limit, axis_limit])
	plotname = "{:06d}.png".format(frame)
	lb.savefig(plotname)

if __name__ == "__main__":
	# Get a list of all the files.
	fnames = [fname for fname in os.listdir(os.path.split(os.path.abspath(__file__))[0]) if re.match(r"positions_(?P<num>\d+)\.csv", fname)]
	# Make the plots.
	for fname in fnames:
		makeplot(fname)
	# Make the movie.
	os.system("ffmpeg -y -i '%06d.png' output.m4v")
	# Clean up the plots.
	os.system("rm *.png")