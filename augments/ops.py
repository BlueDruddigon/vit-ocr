import cv2
import numpy as np
from scipy.ndimage import zoom as sci_zoom
from wand.api import library as wand_libs
from wand.image import Image as WandImage


class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0, channel=None):
        wand_libs.MagickMotionBlurImage(self.wand, radius, sigma, angle)


def clipped_zoom(img, zoom_factor):
    h = img.shape[1]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / float(zoom_factor)))

    top = (h - ch) // 2
    img = sci_zoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # super-sample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(map_size=256, wibble_decay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return a square 2d array, side length 'map_size', of floats in range 0-255.
    'map_size' must be a power of two.
    """
    assert map_size & (map_size - 1) == 0
    map_array = np.empty((map_size, map_size), dtype=np.float_)
    map_array[0, 0] = 0
    step_size = map_size
    wibble = 100

    def wibble_mean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fill_squares():
        """For each square of points step_size apart,
           calculate middle value as mean of points + wibble"""
        corner_ref = map_array[0:map_size:step_size, 0:map_size:step_size]
        square_accum = corner_ref + np.roll(corner_ref, shift=-1, axis=0)
        square_accum += np.roll(square_accum, shift=-1, axis=1)
        map_array[step_size // 2:map_size:step_size, step_size // 2:map_size:step_size] = wibble_mean(square_accum)

    def fill_diamonds():
        """For each diamond of points step_size apart,
           calculate middle value as mean of points + wibble"""
        map_size = map_array.shape[0]
        dr_grid = map_array[step_size // 2:map_size:step_size, step_size // 2:map_size:step_size]
        ul_grid = map_array[0:map_size:step_size, 0:map_size:step_size]
        ldr_sum = dr_grid + np.roll(dr_grid, 1, axis=0)
        lul_sum = ul_grid + np.roll(ul_grid, -1, axis=1)
        lt_sum = ldr_sum + lul_sum
        map_array[0:map_size:step_size, step_size // 2:map_size:step_size] = wibble_mean(lt_sum)
        tdr_sum = dr_grid + np.roll(dr_grid, 1, axis=1)
        tul_sum = ul_grid + np.roll(ul_grid, -1, axis=0)
        tt_sum = tdr_sum + tul_sum
        map_array[step_size // 2:map_size:step_size, 0:map_size:step_size] = wibble_mean(tt_sum)

    while step_size >= 2:
        fill_squares()
        fill_diamonds()
        step_size //= 2
        wibble /= wibble_decay

    map_array -= np.amin(map_array)
    return map_array / np.amax(map_array)
