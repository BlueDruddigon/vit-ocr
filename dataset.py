import glob
import math
import os
import re
import sys

import lmdb
import numpy as np
import six
import torch
from PIL import Image
from natsort import natsorted
from torch.utils.data import Dataset
from torchvision import transforms

from augments import *


class LmdbDataset(Dataset):
    def __init__(self, root, opt) -> None:
        super(LmdbDataset, self).__init__()
        self.root = root
        self.opt = opt
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print(f'Cannot create lmdb from {root}')
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            self.n_samples = int(txn.get('num-samples'.encode()))

            if self.opt.data_filtering_off:
                self.filtered_index_list = [index + 1 for index in range(self.n_samples)]
            else:
                self.filtered_index_list = []
                for index in range(self.n_samples):
                    index += 1  # lmdb start
                    label_key = f'label-{index:09d}'.encode()
                    label = txn.get(label_key).decode('utf-8')

                    if len(label) > self.opt.batch_max_length:
                        continue

                    # By default, images containing characters which are not in opt.character are filtered.
                    # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                    out_of_char = f'[^{self.opt.character}]'
                    if re.search(out_of_char, label.lower()):
                        continue

                    self.filtered_index_list.append(index)
                self.n_samples = len(self.filtered_index_list)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = f'label-{index:09d}'.encode()
            label = txn.get(label_key).decode('utf-8')
            img_key = f'image-{index:09d}'.encode()
            img_buf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(img_buf)
            buf.seek(0)
            try:
                if self.opt.rgb:
                    img = Image.open(buf).convert('RGB')
                else:
                    img = Image.open(buf).convert('L')
            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy input and dummy label for corrupted image
                if self.opt.rgb:
                    img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                else:
                    img = Image.new('L', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'

            if not self.opt.sensitive:
                label = label.lower()

            # We only train and evaluate on alphabet and numeric
            out_of_char = f'[^{self.opt.character}]'
            label = re.sub(out_of_char, '', label)

        return img, label


class RawDataset(Dataset):
    def __init__(self, root, opt) -> None:
        super(RawDataset, self).__init__()
        self.opt = opt
        self.image_path_list = []
        for ext in ['jpg', 'jpeg', 'png']:
            images = glob.glob(os.path.join(root, f'*.{ext}'))
            self.image_path_list.extend(images)

        self.image_path_list = natsorted(self.image_path_list)
        self.n_samples = len(self.image_path_list)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        try:
            if self.opt.rgb:
                img = Image.open(self.image_path_list[index]).convert('RGB')
            else:
                img = Image.open(self.image_path_list[index]).convert('L')
        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image
            if self.opt.rgb:
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))
        return img, self.image_path_list[index]


def is_less(prob=.5):
    return np.random.uniform(0, 1) < prob


class DataAugment:
    def __init__(self, opt) -> None:
        self.opt = opt

        if not opt.eval:
            self.process = [Posterize(), Solarize(), Invert(), Equalize(), AutoContrast(), Sharpness(), Color()]
            self.camera = [Contrast(), Brightness(), JpegCompression(), Pixelate()]
            self.pattern = [VGrid(), HGrid(), Grid(), RectGrid(), EllipseGrid()]
            self.noise = [GaussianNoise(), ShotNoise(), ImpulseNoise(), SpeckleNoise()]
            self.blur = [GaussianBlur(), RefocusBlur(), MotionBlur(), GlassBlur(), ZoomBlur()]
            self.weather = [Fog(), Snow(), Frost(), Rain(), Shadow()]

            self.noises = [self.blur, self.noise, self.weather]
            self.processes = [self.camera, self.process]

            self.warp = [Curve(), Distort(), Stretch()]
            self.geometry = [Rotate(), Perspective(), Shrink()]

            self.is_baseline_aug = False
            # Random augment
            if self.opt.is_random_aug:
                self.augs = [
                    self.process, self.camera, self.noise, self.blur, self.weather, self.pattern, self.warp,
                    self.geometry
                ]
            # Semantic Augment
            elif self.opt.is_semantic_aug:
                self.noise = [GaussianNoise()]
                self.blur = [MotionBlur()]
                self.augs = [self.noise, self.blur, self.geometry]
            # pp-ocr augment
            elif self.opt.is_learning_aug:
                self.geometry = [Rotate(), Perspective()]
                self.noise = [GaussianNoise()]
                self.blur = [MotionBlur()]
                self.warp = [Distort()]
                self.augs = [self.warp, self.noise, self.blur, self.geometry]
                self.is_baseline_aug = True
            # scatter augment
            elif self.opt.is_scatter_aug:
                self.geometry = [Shrink()]
                self.warp = [Distort()]
                self.augs = [self.warp, self.geometry]
                self.is_baseline_aug = True
            # rotation augment
            elif self.opt.is_rotation_aug:
                self.geometry = [Rotate()]
                self.augs = [self.geometry]
                self.is_baseline_aug = True

        self.scale = False

    def __call__(self, img):
        img = img.resize((self.opt.imgW, self.opt.imgH), Image.BICUBIC)

        if self.opt.eval or is_less(self.opt.intact_prob):
            pass
        elif self.opt.is_random_aug or self.is_baseline_aug:
            img = self.random_aug(img)
        elif self.opt.is_select_aug:
            img = self.select_aug(img)

        img = transforms.ToTensor()(img)
        if self.scale:
            img.sub_(0.5).div_(0.5)
        return img

    def random_aug(self, img):
        augs = np.random.choice(self.augs, self.opt.augs_num, replace=False)
        for aug in augs:
            index = np.random.randint(0, len(aug))
            op = aug[index]
            mag = np.random.randint(0, 3) if self.opt.augs_mag is None else self.opt.augs_mag
            if type(op).__name__ == 'Rain' or type(op).__name__ == 'Grid':
                img = op(img.copy(), mag=mag)
            else:
                img = op(img, mag=mag)

        return img

    def select_aug(self, img):
        prob = 1.

        if self.opt.process:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.process))
            op = self.process[index]
            img = op(img, mag=mag, prob=prob)
        if self.opt.noise:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.noise))
            op = self.noise[index]
            img = op(img, mag=mag, prob=prob)
        if self.opt.blur:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.blur))
            op = self.blur[index]
            img = op(img, mag=mag, prob=prob)
        if self.opt.weather:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.weather))
            op = self.weather[index]
            if type(op).__name__ == 'Rain':
                img = op(img.copy(), mag=mag, prob=prob)
            else:
                img = op(img, mag=mag, prob=prob)
        if self.opt.camera:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.camera))
            op = self.camera[index]
            img = op(img, mag=mag, prob=prob)
        if self.opt.pattern:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.pattern))
            op = self.pattern[index]
            img = op(img.copy(), mag=mag, prob=prob)
        is_curve = False
        if self.opt.warp:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.warp))
            op = self.warp[index]
            if type(op).__name__ == 'Curve':
                is_curve = True
            img = op(img, mag=mag, prob=prob)
        if self.opt.geometry:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.geometry))
            op = self.geometry[index]
            if type(op).__name__ == 'Rotate':
                img = op(img, is_curve=is_curve, mag=mag, prob=prob)
            else:
                img = op(img, mag=mag, prob=prob)

        return img


class ResizeNormalize:
    def __init__(self, size, interpolation=Image.BICUBIC) -> None:
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD:
    def __init__(self, max_size, pad_type: str = 'right') -> None:
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.pad_type = pad_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:
            pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
        return pad_img


class AlignCollate:
    def __init__(self, imgH: int = 32, imgW: int = 32, keep_ratio_with_pad: bool = False, opt=None) -> None:
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.opt = opt

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:
            resized_max_w = self.imgW
            input_channels = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channels, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)
        else:
            transform = DataAugment(self.opt)
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels
