import numpy as np
import sys
import math

def load_ppm(filename, verbose=True):
  f = open(filename, 'rb')
  tag = f.readline().decode('utf8')
  channel = 0
  if tag[0:2] == 'P5': #single channel
    channel = 1
  elif tag[0:2] == 'P6': #3-channel
    channel = 3
  else: #not a PPM
    f.close()
    return None

  segments = f.readline().decode('utf8').strip().split(' ')
  width = int(segments[0])
  height = int(segments[1])
  max_val = int(f.readline())

  bit_width = 0
  while max_val != 0:
    max_val >>= 1
    bit_width += 1

  length = width * height * channel * bit_width
  buf = f.read(length)
  f.close()

  if bit_width == 8:
    img = np.frombuffer(buf, dtype=np.uint8)
  elif bit_width == 16:
    img = np.frombuffer(buf, dtype=np.uint16)
  elif bit_width == 32:
    img = np.frombuffer(buf, dtype=np.uint32)
  elif bit_width == 64:
    img = np.frombuffer(buf, dtype=np.uint64)

  img = img.reshape([height, width, -1])
  if verbose:
    print('Read image %s.' % (filename))
  return img

def write_ppm(img, filename, verbose=True):
  f = open(filename, 'wb')

  if img.ndim == 2:
    f.write(('P5\n').encode())
  elif img.ndim == 3:
    if img.shape[2] == 1:
      f.write(('P5\n').encode())
      img = img[:, :, 0]
    else:
      f.write(('P6\n').encode())
  height, width = img.shape[0:2]
  f.write((str(width) + ' ' + str(height) + '\n' + str(np.iinfo(img.dtype).max) + '\n').encode())
  f.write(memoryview(img))

  if verbose:
    print('Wrote to %s' % (filename))
  f.close()

if __name__ == '__main__':
  pass



