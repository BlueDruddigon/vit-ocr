import numpy as np
import skimage as sk
from PIL import Image


class GaussianNoise:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        b = [.08, 0.1, 0.12]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag
        a = b[index]
        c = np.random.uniform(a, a + 0.03)
        img = np.array(img) / 255.
        img = np.clip(img + np.random.normal(size=img.shape, scale=c), 0, 1) * 255
        return Image.fromarray(img.astype(np.uint8))


class ShotNoise:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        b = [13, 8, 3]
        if mag < 0 or mag >= len(b):
            index = 2
        else:
            index = mag
        a = b[index]
        c = np.random.uniform(a, a + 7)
        img = np.array(img) / 255.
        img = np.clip(np.random.poisson(img * c) / float(c), 0, 1) * 255
        return Image.fromarray(img.astype(np.uint8))


class ImpulseNoise:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        b = [.03, .07, .11]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag
        a = b[index]
        c = np.random.uniform(a, a + .04)
        img = sk.util.random_noise(np.array(img) / 255., mode='s&p', amount=c) * 255
        return Image.fromarray(img.astype(np.uint8))


class SpeckleNoise:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        b = [.15, .2, .25]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag
        a = b[index]
        c = np.random.uniform(a, a + .05)
        img = np.array(img) / 255.
        img = np.clip(img + img * np.random.normal(size=img.shape, scale=c), 0, 1) * 255
        return Image.fromarray(img.astype(np.uint8))
