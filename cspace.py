import numpy as np

def RGB2XYZ(rgbs):  # size of [3, n]
    mat = np.array([
        [0.49, 0.31, 0.2],
        [0.17697, 0.8124, 0.01063],
        [0, 0.01, 0.99]]) * (1 / 0.17697)
    return np.dot(mat, rgbs)

def XYZ2xyY(XYZs):  # size of [3, n]
    Ys = XYZs[1, :]
    s = sum(XYZs, 0)
    return np.vstack([(XYZs / s)[:2, :], Ys])

def xyY2XYZ(xyYs):  # size of [3, n]
    Xs = xyYs[0, :] / xyYs[1, :] * xyYs[2, :]
    Zs = (-xyYs[0, :] - xyYs[1, :] + 1) / xyYs[1, :] * xyYs[2, :]
    return np.vstack([Xs, xyYs[2, :], Zs])

def XYZ2RGB(XYZs):
    mat = np.array([
        [0.41847, -0.15866, -0.082835],
        [-0.091169, 0.25243, 0.015708],
        [0.00092090, -0.0025498, 0.1786]])
    return np.dot(mat, XYZs)

if __name__ == '__main__':
    pass