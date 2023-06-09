import numpy as np
from PIL import ImageEnhance, ImageOps


class Posterize:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        c = [1, 3, 6]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]
        bit = np.random.randint(c, c + 2)
        img = ImageOps.posterize(img, bit)

        return img


class Solarize:
    def __init__(self):
        pass

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        c = [64, 128, 192]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]
        thresh = np.random.randint(c, c + 64)
        img = ImageOps.solarize(img, thresh)

        return img


class Invert:
    def __call__(self, img, _=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        img = ImageOps.invert(img)

        return img


class Equalize:
    def __call__(self, img, _=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        img = ImageOps.equalize(img)

        return img


class AutoContrast:
    def __call__(self, img, _=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        img = ImageOps.autocontrast(img)

        return img


class Sharpness:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        c = [.1, .7, 1.3]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]
        magnitude = np.random.uniform(c, c + .6)
        img = ImageEnhance.Sharpness(img).enhance(magnitude)

        return img


class Color:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        c = [.1, .7, 1.3]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]
        magnitude = np.random.uniform(c, c + .6)
        img = ImageEnhance.Color(img).enhance(magnitude)

        return img
