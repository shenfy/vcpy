import numpy as np
import sys

def load_pfm(filename, verbose=True):
	FLOATSIZE = 4;

	f = open(filename, 'rb')
	tag = f.readline().decode('utf8')

	line = f.readline().decode('utf8')
	segments = line.split(' ')
	width = int(segments[0])
	height = int(segments[1])
	line = f.readline() #-1.000000 discard

	if tag[0:2] == 'Pf': #single channel
		length = width * height * FLOATSIZE
	elif tag[0:2] == 'PF': #3-channel
		length = width * height * 3 * FLOATSIZE;
	else: #not a PFM
		f.close()
		return None

	buf = f.read(length)
	f.close()

	#convert to numpy data structure
	img = np.frombuffer(buf, dtype=np.float32)
	img = img.reshape([height, width, -1])
	img = np.flipud(img)
	if verbose:
		print("Read image %s." % (filename))
	return img

def write_pfm(img, filename, verbose=True):
	FLOATSIZE = 4;

	f = open(filename, 'wb')

	if img.ndim == 2: #single channel
		height, width = img.shape
		tmpimg = np.flipud(img).reshape(height * width).astype(np.float32)
		f.write(("Pf\n" + str(width) + ' ' + str(height) + '\n').encode())
		f.write("-1.00000\n".encode())
		f.write(memoryview(tmpimg))
		if verbose:
			print("Wrote to %s" % (filename))

	elif img.ndim == 3: #color image
		if img.shape[2] == 1:
			img3 = np.tile(img, [1, 1, 3])
		else:
			img3 = img
		height, width = img3.shape[0:2]
		tmpimg = np.flipud(img3).reshape(height * width * 3).astype(np.float32)
		f.write(("PF\n" + str(width) + ' ' + str(height) + '\n').encode())
		f.write("-1.00000\n".encode())
		f.write(memoryview(tmpimg))
		if verbose:
			print("Wrote to %s" % (filename))

	f.close()

if __name__ == '__main__':
	pass