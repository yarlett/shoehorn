import csv, numpy as np, os, pylab as lb, sys

symbols = {
	"0": ".",
	"1": "o",
	"2": "v",
	"3": "^",
	"4": "+",
	"5": "x",
	"6": "d",
	"7": "*",
	"8": "s",
	"9": "h",
}

if __name__ == "__main__":
	# Find the most recently modified CSV file matching the prefix in sys.argv[1].
	directory, prefix = os.path.split(sys.argv[1])
	directory = os.path.abspath(directory)
	# Get a list of matching files.
	max_details = {"modtime": None, "filename": None}
	for filename in os.listdir(directory):
		if filename[:len(prefix)] == prefix:
			modtime = os.path.getmtime(os.path.join(directory, filename))
			if max_details["modtime"] is None or modtime > max_details["modtime"]:
				max_details["modtime"] = modtime
				max_details["filename"] = os.path.join(directory, filename)
	# Load the data.
	D = {}
	f = open(max_details["filename"], "r")
	reader = csv.reader(f)
	for row in reader:
		obj_name = row[0][0]	
		if obj_name not in D:
			D[obj_name] = []
		D[obj_name].append([float(v) for v in row[1:]])
	# Plot the object positions.
	fig = lb.figure()
	for n in sorted(D):
		dat = np.array(D[n], "d")
		lb.plot(dat[:, 0], dat[:, 1], symbols[n], label="{:s}".format(n))
	# lb.legend(loc=0)
	lb.title("Positions from {:s}".format(max_details["filename"]), size=12.0)
	lb.axis("equal")
	lb.show()