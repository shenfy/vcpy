from numba import jit
from pylab import *

# 3d math helper
@jit
def magnitude(v):
    return sqrt(sum(v ** 2))

@jit
def normalize(v):
    m = magnitude(v)
    if m == 0:
        return v
    return v / m

@jit
def translate(xyz):
    x, y, z = xyz
    return array([[1, 0, 0, x],
                  [0, 1, 0, y],
                  [0, 0, 1, z],
                  [0, 0, 0, 1]])

@jit
def scale(xyz):
    x, y, z = xyz
    return array([[x, 0, 0, 0],
                  [0, y, 0, 0],
                  [0, 0, z, 0],
                  [0, 0, 0, 1]])

@jit
def gl_lookat(origin, target, up):
    F = target[:3] - origin[:3]
    f = normalize(F)
    U = normalize(up[:3])
    s = normalize(cross(f, U))
    u = cross(s, f)
    M = identity(4)
    M[:3,:3] = np.vstack([s,u,-f])
    T = translate(-origin[:3])
    return dot(M, T)

@jit
def gl_frustum(left, right, bottom, top, near, far):
    inv_fmn = 1.0 / (far - near)
    inv_tmb = 1.0 / (top - bottom)
    inv_rml = 1.0 / (right - left)

    return array([
        [2 * near * inv_rml, 0, (right + left) * inv_rml, 0],
        [0, 2 * near * inv_tmb, (top + bottom) * inv_tmb, 0],
        [0, 0, -(far + near) * inv_fmn, -2 * far * near * inv_fmn],
        [0, 0, -1, 0]])

@jit
def mitsuba_frustum(left, right, bottom, top, near, far):
    inv_fmn = 1.0 / (far - near)
    inv_tmb = 1.0 / (top - bottom)
    inv_rml = 1.0 / (right - left)

    return array([
        [2 * near * inv_rml, 0, (right + left) * inv_rml, 0],
        [0, 2 * near * inv_tmb, -(top + bottom) * inv_tmb, 0],
        [0, 0, far * inv_fmn, -far * near * inv_fmn],
        [0, 0, 1, 0]])

@jit
def gl_perspective(fov, aspect_ratio, near, far):
    recip = 1.0 / (near - far)
    cotan = 1.0 / tan(deg2rad(fov / 2.0))

    return array([
        [cotan / aspect_ratio, 0, 0, 0],
        [0, cotan, 0, 0],
        [0, 0, (near + far) * recip, 2 * far * near * recip],
        [0, 0, -1, 0]])

@jit
def mitsuba_perspective(fov, near, far):
    recip = 1.0 / (far - near)
    cotan = 1.0 / tan(deg2rad(fov / 2.0))

    return array([
        [cotan, 0, 0, 0],
        [0, cotan, 0, 0],
        [0, 0, far * recip, -far * near * recip],
        [0, 0, 1, 0]])

@jit
def gl_viewport(width, height):
    return array([
        [width / 2, 0, 0, (width - 1) / 2],
        [0, height / 2, 0, (height - 1) / 2],
        [0, 0, 0.5, 0.5],
        [0, 0, 0, 1]])

@jit
def v3v4(v3, w):
    return array([v3[0], v3[1], v3[2], w])

def rotation_from_normal(z):
    x = np.array([1, 0, 0])
    if np.dot(x, z) > 0.8:
        x = np.array([0, 1, 0])
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.column_stack([x, y, z])

if __name__ == '__main__':
    pass