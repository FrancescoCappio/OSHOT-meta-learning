# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random
from PIL import Image, ImageOps, ImageFilter
import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string



class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target is None:
            return image
        target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.vflip(image)
            target = target.transpose(1)
        return image, target

class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target


class ToCV2Image(object):
    def __call__(self, tensor, targets=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), targets


class DeNormalize(object):

    def __init__(self, mean, std, reshape=False):
        """
        if reshape:
            self.mean = torch.Tensor(mean)
            self.std = torch.Tensor(std)
        else:
        """
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.reshape = reshape

    def __call__(self, tensor):

        if self.reshape:
            tensor, _ = (ToCV2Image())(tensor)
        tensor = tensor*self.std
        tensor = tensor+self.mean
        if self.reshape:
            tensor, _, = (ToTensor())(tensor, None)
        return tensor


class ToPixelDomain(object):
    def __call__(self, image):
        image = image * 255
        return image

class RandomChannelsExchange(object):
    def __call__(self, image):

        image_result = None
        #r, g, b = image.split()
        #channels_list = [r, g, b]

        #seed = random.randint(0, 5)
        #if seed == 3:
        image_result = ImageOps.invert(image)
        #else:
        #   r_seed = random.randint(0,2)
        #   g_seed = random.randint(0,2)
        #   b_seed = random.randint(0,2)

        #    image_result = Image.merge('RGB', (channels_list[r_seed], channels_list[g_seed], channels_list[b_seed]))

        return image_result

class GaussianBlur(object):
    def __init__(self, radius):
        self.radius = radius


    def __call__(self, image):

        image_result = None

        image_result = image.filter(ImageFilter.GaussianBlur(radius=self.radius))

        return image_result
