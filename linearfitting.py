import math
import numpy as np
from scipy.optimize import lsq_linear
from vcpy.quat import Quat

def fit_plane(pts):
  pts = np.column_stack([pts, np.ones(pts.shape[0], dtype=pts.dtype)])
  _, _, vh = np.linalg.svd(pts, full_matrices=False)
  return vh[-1, :]

def fit_circle(pts):
  pts = np.column_stack([pts, np.ones(pts.shape[0], dtype=pts.dtype)])
  b = np.sum(pts[:, :2] ** 2, axis=1)
  circle = lsq_linear(pts, b).x
  circle[0] /= 2.0
  circle[1] /= 2.0
  radius = np.sqrt(circle[0] ** 2 + circle[1] ** 2 + circle[2])
  return circle[:2], radius

def fit_circle3(pts):
  # fit a plane from these points
  plane = fit_plane(pts)
  # select a arbitary point on the plane as the plane frame origin
  origin = np.zeros_like(plane[:3])
  if abs(plane[3]) > 1e-6:
    plane = plane[:3] / plane[3]
    i = np.argmax(abs(plane))
    origin[i] = -1.0 / plane[i]
  else:
    plane = plane[:3]
  # build the plane orientation
  normal = plane / np.linalg.norm(plane)
  y = np.array([1, 0, 0], dtype=normal.dtype)
  if np.dot(y, normal) > 0.9:
    y = np.array([0, 1, 0], dtype=normal.dtype)
  x = np.cross(y, normal)
  x = x / np.linalg.norm(x)
  y = np.cross(normal, x)
  plane_frame = np.column_stack((x, y, normal))
  # transform points to plane frame, so they near the z = 0 plane
  plane_frame_points = (pts - origin) @ plane_frame
  # fit a circle on xy plane to the points' projections
  center, radius = fit_circle(plane_frame_points[:, :2])
  # return the coordinates in points' space
  center = plane_frame @ np.array([center[0], center[1], 0], dtype=center.dtype) + origin
  return center, normal, radius

def frame_offset(src_frames, dst_frames):
  src_quats = [Quat.from_mat(m[:3, :3]) for m in src_frames]
  dst_quats = [Quat.from_mat(m[:3, :3]) for m in dst_frames]
  delta_quats = [d * Quat.inv(s) for s, d in zip(src_quats, dst_quats)]
  for q in delta_quats:
    if q.v[0] < 0:
      q.scale(-1)

  delta_quats = [q.v for q in delta_quats]
  delta_quat = Quat(np.mean(delta_quats, axis=0))
  delta_quat = Quat.normalize(delta_quat)

  src_positions = [m[:, 3] for m in src_frames]
  dst_positions = [m[:, 3] for m in dst_frames]
  rotated_positions = [delta_quat.apply(p) for p in src_positions]
  delta_translation = np.mean(np.array(dst_positions) - np.array(rotated_positions), axis=0)

  result = delta_quat.to_mat()
  result[:3, 3] = delta_translation[:3]
  return result

def absolute_orientation(src_pts, dst_pts, scale=None):
  src_centroid = np.mean(src_pts, axis=0)
  dst_centroid = np.mean(dst_pts, axis=0)
  src_pts = src_pts - src_centroid
  dst_pts = dst_pts - dst_centroid

  if scale is None:
    scale = math.sqrt(np.sum(dst_pts ** 2) / np.sum(src_pts ** 2))

  M = np.zeros((3, 3), dtype=src_pts.dtype)
  for src_pt, dst_pt in zip(src_pts, dst_pts):
    M += src_pt[..., np.newaxis] @ dst_pt[np.newaxis, ...]

  N = np.array([
    [M[0, 0] + M[1, 1] + M[2, 2], M[1, 2] - M[2, 1], M[2, 0] - M[0, 2], M[0, 1] - M[1, 0]],
    [M[1, 2] - M[2, 1], M[0, 0] - M[1, 1] - M[2, 2], M[0, 1] + M[1, 0], M[2, 0] + M[0, 2]],
    [M[2, 0] - M[0, 2], M[0, 1] + M[1, 0], M[1, 1] - M[0, 0] - M[2, 2], M[1, 2] + M[2, 1]],
    [M[0, 1] - M[1, 0], M[2, 0] + M[0, 2], M[1, 2] + M[2, 1], M[2, 2] - M[0, 0] - M[1, 1]]
  ])
  eigvals, eigvecs = np.linalg.eigh(N)
  rotation = Quat(eigvecs[:, 3])

  rotated = rotation.apply(np.array([*src_centroid, 1], dtype=src_centroid.dtype))[:3]
  translation = dst_centroid - scale * rotated
  return (translation, rotation, scale)

# https://github.com/cjekel/cjekel.github.io/blob/master/assets/2015-09-13/demo.py
# fit a sphere to given points
# returns the radius and center points of
# the best fit sphere
def sphere_fit(pts):
    #   Assemble the A matrix
    spX = np.array(pts)[:, 0]
    spY = np.array(pts)[:, 1]
    spZ = np.array(pts)[:, 2]
    A = np.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residules, rank, singval = np.linalg.lstsq(A,f, rcond=None)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = np.sqrt(t)

    return radius, C[0], C[1], C[2]
