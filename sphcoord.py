from numba import jit
import numpy as np

@jit
def cart2pol(pos):
    x = pos[0]
    y = pos[1]
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return np.array([theta, rho])

@jit
def pol2cart(sph):
    theta = sph[0]
    rho = sph[1]
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return np.array([x, y])

@jit
def cart2sph(pos):
    x = pos[0]
    y = pos[1]
    z = pos[2]
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return np.array([el, az, r])

@jit
def sph2cart(sph):
    el = sph[0]
    az = sph[1]
    r = sph[2]
    r_cos_el = r * np.cos(el)
    x = r_cos_el * np.cos(az)
    y = r_cos_el * np.sin(az)
    z = r * np.sin(el)
    return np.array([x, y, z])

def cart2sph_v(pts_3d):
  hxy = np.hypot(pts_3d[:, 0], pts_3d[:, 1])
  r = np.hypot(hxy, pts_3d[:, 2])
  el = np.arctan2(pts_3d[:, 2], hxy)
  az = np.arctan2(pts_3d[:, 1], pts_3d[:, 0])
  return np.vstack((el, az, r)).T

def sph2cart_v(pts_sph):
  r_cos_el = pts_sph[:, 2] * np.cos(pts_sph[:, 0])
  x = r_cos_el * np.cos(pts_sph[:, 1])
  y = r_cos_el * np.sin(pts_sph[:, 1])
  z = pts_sph[:, 2] * np.sin(pts_sph[:, 0])
  return np.vstack((x, y, z)).T


if __name__ == '__main__':
    pass