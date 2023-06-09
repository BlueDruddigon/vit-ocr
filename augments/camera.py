from io import BytesIO

import numpy as np
import skimage as sk
from PIL import Image, ImageOps


class Contrast:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        c = [0.4, .3, .2]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]
        img = np.array(img) / 255.
        means = np.mean(img, axis=(0, 1), keepdims=True)
        img = np.clip((img - means) * c + means, 0, 1) * 255

        return Image.fromarray(img.astype(np.uint8))


class Brightness:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        c = [.1, .2, .3]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]

        n_channels = len(img.getbands())
        is_gray = n_channels == 1

        img = np.array(img) / 255.
        if is_gray:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)

        img = sk.color.rgb2hsv(img)
        img[:, :, 2] = np.clip(img[:, :, 2] + c, 0, 1)
        img = sk.color.hsv2rgb(img)

        img = np.clip(img, 0, 1) * 255
        img = Image.fromarray(img.astype(np.uint8))
        if is_gray:
            img = ImageOps.grayscale(img)

        return img


class JpegCompression:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        c = [25, 18, 15]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]
        output = BytesIO()
        img.save(output, 'JPEG', quality=c)
        return Image.open(output)


class Pixelate:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        W, H = img.size
        c = [0.6, 0.5, 0.4]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]
        img = img.resize((int(W * c), int(H * c)), Image.BOX)
        return img.resize((W, H), Image.BOX)
