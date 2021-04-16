import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F 
from collections import OrderedDict
import torchvision.transforms as T
import time

def get_color_distortion(s=1):
    color_jitter = torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter =  torchvision.transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray =  torchvision.transforms.RandomGrayscale(p=0.2)
    color_distort =  torchvision.transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

def generate_pairs(images):
    images1 , images2 = [], []
    for img in images:
        img = torchvision.transforms.ToPILImage()(img)
        img1 = get_color_distortion()(img)
        img2 = get_color_distortion()(img)
        images1.append(torchvision.transforms.ToTensor()(img1))
        images2.append(torchvision.transforms.ToTensor()(img2))
    images1 , images2 = torch.stack(images1), torch.stack(images2)
    return images1, images2

def get_aug():
    transform = T.Compose([
            T.RandomResizedCrop((96,96), scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            #T.RandomApply([T.GaussianBlur(kernel_size=96//20*2+1, sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor()
        ])
    return transform

def generate_pairs_simsiam(images):
    images1 , images2 = [], []
    for img in images:
        img = T.ToPILImage()(img)
        img1 = get_aug()(img)
        img2 = get_aug()(img)
        images1.append(img1)
        images2.append(img2)
    images1 , images2 = torch.stack(images1), torch.stack(images2)
    return images1, images2

import torch
import numpy as np


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img