import numpy as np
from PIL import Image
import random
import cv2
import torch
#from torchvision import transforms as T
#from torchvision.transforms import functional as F
from opencv_transforms import functional as F
from opencv_transforms import transforms as T
from scipy.spatial.transform import Rotation as R
import albumentations as A
from albumentations import functional as FA


# def pad_if_smaller(img, size, fill=0):
#     min_size = min(img.size)
#     if min_size < size:
#         ow, oh = img.size
#         padh = size - oh if oh < size else 0
#         padw = size - ow if ow < size else 0
#         img = F.pad(img, (0, 0, padw, padh), fill=fill)
#     return img

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, image2):
        for t in self.transforms:
            image, image2 = t(image, image2)
        return image, image2



class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, image2):
        image = F.center_crop(image, self.size)
        image2 = F.center_crop(image2, self.size)
        return image, image2


class Resize(object): 
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = (size,size) #TODO according to what is size
        self.interpolation = interpolation

    def __call__(self, image, image2): #TODO warning for mask -> nearest interpolation?
        image = F.resize(image, self.size, self.interpolation)
        image2 = F.resize(image2, self.size, self.interpolation)
        return image, image2


class ToTensor(object):
    def __call__(self, image, image2):
        image = F.to_tensor(image)
        image2 = F.to_tensor(image2)
        #image2 = torch.as_tensor(np.array(image2), dtype=torch.int64) #for masks
        return image, image2


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, image2):
        image = F.normalize(image, mean=self.mean, std=self.std)
        image2 = F.normalize(image2, mean=self.mean, std=self.std)
        return image, image2


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, image2):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            image2 = F.hflip(image2) 
        return image, image2

class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, image2):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            image2 = F.vflip(image2)
        return image, image2

class Random90DegRotAntiClockWise(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, image2):
        if random.random() < self.flip_prob:
            angle = -90
            image =  F.rotate(image, angle, resample=False, expand=False, center=None)
            image2 =  F.rotate(image2, angle, resample=False, expand=False, center=None)
        return image, image2

class Random90DegRotClockWise(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, image2):
        if random.random() < self.flip_prob:
            angle = 90
            image =  F.rotate(image, angle, resample=False, expand=False, center=None)
            image2 =  F.rotate(image2, angle, resample=False, expand=False, center=None)
        return image, image2

class Random180DegRot(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, image2):
        if random.random() < self.flip_prob:
            angle = 180
            image =  F.rotate(image, angle, resample=False, expand=False, center=None)
            image2 =  F.rotate(image2, angle, resample=False, expand=False, center=None)
        return image, image2

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, image2):
        #image = pad_if_smaller(image, self.size)
        #image2 = pad_if_smaller(image2, self.size)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        image2 = F.crop(image2, *crop_params)
        return image, image2



class RandomResize(object):
    def __init__(self, low, high, interpolation=Image.BILINEAR):
        self.low = low
        self.high = high
        self.interpolation = interpolation

    def __call__(self, image, image2): 
        size = np.random.randint(self.low, self.high)
        image = F.resize(image, (size,size), self.interpolation)
        image2 = F.resize(image2, (size,size), self.interpolation)
        return image, image2


class RandomRotation(object):
    def __init__(self, degrees, resample=False, expand=False, center=None, fill=None):
        self.degrees = degrees
        self.expand = expand
        self.resample = resample
        self.center = center
        self.fill = fill

    def __call__(self, image, image2): 
        angle = T.RandomRotation.get_params(self.degrees)
        image =  F.rotate(image, angle, self.resample, self.expand, self.center)
        image2 =  F.rotate(image2, angle, self.resample, self.expand, self.center )

        return image, image2

class Albumentations(object):
    def __init__(self, hue_limit, sat_limit, flip_prob):
        self.hue_limit = hue_limit
        self.sat_limit = sat_limit
        self.flip_prob=flip_prob

    def __call__(self, image, normals):
        if random.random() < self.flip_prob:
            hue_shift=random.uniform(-self.hue_limit, self.hue_limit)
            sat_shift=random.uniform(-self.sat_limit, self.sat_limit)
            image = FA.shift_hsv(image, hue_shift=hue_shift,sat_shift=sat_shift, val_shift=0)
        return image, normals
