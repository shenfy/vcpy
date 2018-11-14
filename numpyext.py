import numpy as np

def to_homo(m, transpose=False):
  if transpose:
    return to_homo(m.T).T
  assert m.ndim == 2
  return np.vstack([m, np.ones((1, m.shape[1]), dtype=m.dtype)])

def from_homo(m, transpose=False):
  if transpose:
    return from_homo(m.T).T
  assert m.ndim == 2
  return m[:-1, :] / m[-1, :]
