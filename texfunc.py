from numba import jit
import numpy as np
from PIL import Image

# bilinear interpolation
@jit
def texture2D(img, uv):
    h, w, c = img.shape
    img_size = np.array([w, h])
    texel = np.array([uv[0], 1.0 - uv[1]]) * img_size - 0.5

    lxy = np.floor(texel)
    dx, dy = texel - lxy
    x0, y0 = lxy
    x1, y1 = lxy + 1

    f00 = img[int(y0), int(x0), :]
    f10 = img[int(y0), int(x1), :]
    f01 = img[int(y1), int(x0), :]
    f11 = img[int(y1), int(x1), :]

    return f00 * (1 - dx) * (1 - dy) + f10 * dx * (1 - dy)\
        + f01 * (1 - dx) * dy + f11 * dx * dy

@jit
def texture2D_px(img, xy):
    h, w, c = img.shape
    img_size = np.array([w, h])

    lxy = np.floor(xy)
    dx, dy = xy - lxy
    x0, y0 = lxy
    x1, y1 = lxy + 1

    f00 = img[int(y0), int(x0), :]
    f10 = img[int(y0), int(x1), :]
    f01 = img[int(y1), int(x0), :]
    f11 = img[int(y1), int(x1), :]

    return f00 * (1 - dx) * (1 - dy) + f10 * dx * (1 - dy)\
        + f01 * (1 - dx) * dy + f11 * dx * dy

@jit
def texture2D_nn(img, uv):
    h, w, c = img.shape
    img_size = np.array([w, h])
    texel = uv * img_size - 0.5

    xy = np.round(texel)
    return img[int(xy[1]), int(xy[0]), :]

if __name__ == '__main__':
    pass