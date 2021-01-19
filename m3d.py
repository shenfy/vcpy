import numpy as np
from numba import njit

# 3d math helper
@njit
def normalize(v):
    m = np.linalg.norm(v)
    if m == 0:
        return v
    return v / m

@njit
def translate(xyz):
    x, y, z = xyz
    return np.array([[1.0, 0.0, 0.0, x],
                  [0.0, 1.0, 0.0, y],
                  [0.0, 0.0, 1.0, z],
                  [0.0, 0.0, 0.0, 1.0]])

@njit
def scale(xyz):
    x, y, z = xyz
    return np.array([[x, 0.0, 0.0, 0.0],
                  [0.0, y, 0.0, 0.0],
                  [0.0, 0.0, z, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])

@njit
def gl_lookat(origin, target, up):
    F = target[:3] - origin[:3]
    f = normalize(F)
    U = normalize(up[:3])
    s = normalize(np.cross(f, U))
    u = np.cross(s, f)
    M = np.identity(4)
    M[:3,:3] = np.vstack((s,u,-f))
    T = translate(-origin[:3])
    return np.dot(M, T)

@njit
def gl_frustum(left, right, bottom, top, near, far):
    inv_fmn = 1.0 / (far - near)
    inv_tmb = 1.0 / (top - bottom)
    inv_rml = 1.0 / (right - left)

    return np.array([
        [2 * near * inv_rml, 0.0, (right + left) * inv_rml, 0.0],
        [0.0, 2 * near * inv_tmb, (top + bottom) * inv_tmb, 0.0],
        [0.0, 0.0, -(far + near) * inv_fmn, -2 * far * near * inv_fmn],
        [0.0, 0.0, -1.0, 0.0]])

@njit
def mitsuba_frustum(left, right, bottom, top, near, far):
    inv_fmn = 1.0 / (far - near)
    inv_tmb = 1.0 / (top - bottom)
    inv_rml = 1.0 / (right - left)

    return np.array([
        [2 * near * inv_rml, 0.0, (right + left) * inv_rml, 0.0],
        [0.0, 2 * near * inv_tmb, -(top + bottom) * inv_tmb, 0.0],
        [0.0, 0.0, far * inv_fmn, -far * near * inv_fmn],
        [0.0, 0.0, 1.0, 0.0]])

@njit
def gl_perspective(fov, aspect_ratio, near, far):
    recip = 1.0 / (near - far)
    cotan = 1.0 / np.tan(np.deg2rad(fov / 2.0))

    return np.array([
        [cotan / aspect_ratio, 0.0, 0.0, 0.0],
        [0.0, cotan, 0.0, 0.0],
        [0.0, 0.0, (near + far) * recip, 2 * far * near * recip],
        [0.0, 0.0, -1.0, 0.0]])

@njit
def mitsuba_perspective(fov, near, far):
    recip = 1.0 / (far - near)
    cotan = 1.0 / np.tan(np.deg2rad(fov / 2.0))

    return np.array([
        [cotan, 0.0, 0.0, 0.0],
        [0.0, cotan, 0.0, 0.0],
        [0.0, 0.0, far * recip, -far * near * recip],
        [0.0, 0.0, 1.0, 0.0]])

@njit
def gl_viewport(width, height):
    return np.array([
        [width / 2, 0.0, 0.0, (width - 1) / 2],
        [0.0, height / 2, 0.0, (height - 1) / 2],
        [0.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 0.0, 1.0]])

@njit
def v3v4(v3, w):
    return np.array([v3[0], v3[1], v3[2], w])

@njit
def rotation_from_normal(z):
    x = np.array([1.0, 0.0, 0.0], dtype=z.dtype)
    if np.dot(x, z) > 0.8:
        x = np.array([0.0, 1.0, 0.0], dtype=z.dtype)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.column_stack((x, y, z))

if __name__ == '__main__':
    pass
