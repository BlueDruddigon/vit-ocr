import cv2
import numpy as np
from PIL import Image


class Shrink:
    def __init__(self):
        self.tps = cv2.createThinPlateSplineShapeTransformer()
        self.translateXAbs = TranslateXAbs()
        self.translateYAbs = TranslateYAbs()

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        W, H = img.size
        img = np.array(img)
        src_points = list()
        dst_points = list()

        W_33 = 0.33 * W
        W_66 = 0.66 * W

        H_50 = 0.50 * H
        P = 0

        b = [.2, .3, .4]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag
        frac = b[index]

        # left-most
        src_points.append([P, P])
        src_points.append([P, H - P])
        x = np.random.uniform(frac - .1, frac) * W_33
        y = np.random.uniform(frac - .1, frac) * H_50
        dst_points.append([P + x, P + y])
        dst_points.append([P + x, H - P - y])

        # 2nd left-most
        src_points.append([P + W_33, P])
        src_points.append([P + W_33, H - P])
        dst_points.append([P + W_33, P + y])
        dst_points.append([P + W_33, H - P - y])

        # 3rd left-most
        src_points.append([P + W_66, P])
        src_points.append([P + W_66, H - P])
        dst_points.append([P + W_66, P + y])
        dst_points.append([P + W_66, H - P - y])

        # right-most
        src_points.append([W - P, P])
        src_points.append([W - P, H - P])
        dst_points.append([W - P - x, P + y])
        dst_points.append([W - P - x, H - P - y])

        N = len(dst_points)
        matches = [cv2.DMatch(i, i, 0) for i in range(N)]
        dst_shape = np.array(dst_points).reshape((-1, N, 2))
        src_shape = np.array(src_points).reshape((-1, N, 2))
        self.tps.estimateTransformation(dst_shape, src_shape, matches)
        img = self.tps.warpImage(img)
        img = Image.fromarray(img)

        if np.random.uniform(0, 1) < 0.5:
            img = self.translateXAbs(img, val=x)
        else:
            img = self.translateYAbs(img, val=y)

        return img


class Rotate:
    def __init__(self, square_side=224):
        self.side = square_side

    def __call__(self, img, is_curve=False, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        W, H = img.size

        if H != self.side or W != self.side:
            img = img.resize((self.side, self.side), Image.BICUBIC)

        b = [20., 40, 60]
        if mag < 0 or mag >= len(b):
            index = 1
        else:
            index = mag
        rotate_angle = b[index]

        angle = np.random.uniform(rotate_angle - 20, rotate_angle)
        if np.random.uniform(0, 1) < 0.5:
            angle = -angle

        expand = False if is_curve else True
        img = img.rotate(angle=angle, resample=Image.BICUBIC, expand=expand)
        img = img.resize((W, H), Image.BICUBIC)

        return img


class Perspective:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        W, H = img.size

        # upper-left, upper-right, lower-left, lower-right
        src = np.float32([[0, 0], [W, 0], [0, H], [W, H]])

        b = [.1, .2, .3]
        if mag < 0 or mag >= len(b):
            index = 2
        else:
            index = mag
        low = b[index]

        high = 1 - low
        if np.random.uniform(0, 1) > 0.5:
            top_right_Y = np.random.uniform(low, low + .1) * H
            bottom_right_Y = np.random.uniform(high - .1, high) * H
            dest = np.float32([[0, 0], [W, top_right_Y], [0, H], [W, bottom_right_Y]])
        else:
            top_left_Y = np.random.uniform(low, low + .1) * H
            bottom_left_Y = np.random.uniform(high - .1, high) * H
            dest = np.float32([[0, top_left_Y], [W, 0], [0, bottom_left_Y], [W, H]])
        M = cv2.getPerspectiveTransform(src, dest)
        img = np.array(img)
        img = cv2.warpPerspective(img, M, (W, H))
        img = Image.fromarray(img)

        return img


class TranslateX:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        b = [.03, .06, .09]
        if mag < 0 or mag >= len(b):
            index = 2
        else:
            index = mag
        v = b[index]
        v = np.random.uniform(v - 0.03, v)

        v = v * img.size[0]
        if np.random.uniform(0, 1) > 0.5:
            v = -v
        return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


class TranslateY:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        b = [.07, .14, .21]
        if mag < 0 or mag >= len(b):
            index = 2
        else:
            index = mag
        v = b[index]
        v = np.random.uniform(v - 0.07, v)

        v = v * img.size[1]
        if np.random.uniform(0, 1) > 0.5:
            v = -v
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


class TranslateXAbs:
    def __call__(self, img, val=0, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        v = np.random.uniform(0, val)

        if np.random.uniform(0, 1) > 0.5:
            v = -v
        return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


class TranslateYAbs:
    def __call__(self, img, val=0, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        v = np.random.uniform(0, val)

        if np.random.uniform(0, 1) > 0.5:
            v = -v
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))
