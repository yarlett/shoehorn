import csv, numpy as np, pylab as lb

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

if __name__ == '__main__':
	for i in xrange(1, 2):
		D = {}
		f = open('mnist_locations{:d}.csv'.format(i), 'r')
		reader = csv.reader(f)
		for row in reader:
			obj_name = row[0][0]	
			if obj_name not in D:
				D[obj_name] = []
			D[obj_name].append([float(v) for v in row[1:]])

		fig = lb.figure()
		for n in sorted(D):
			dat = np.array(D[n], 'd')
			lb.plot(dat[:, 0], dat[:, 1], symbols[n], label='{}'.format(n))
		# lb.legend(loc=0)
		lb.title("Positions {:d}".format(i))
		lb.axis("equal")
	lb.show()