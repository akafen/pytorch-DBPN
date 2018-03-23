# -*- coding: utf-8 -*-
import os, tarfile, random
import sys
from six.moves import urllib
from torchvision.transforms import Compose, CenterCrop, RandomCrop, ToTensor, Scale, RandomHorizontalFlip
from dataset import DatasetFromFolder
from PIL import Image


CROP_SIZE = 256


class RandomRotate(object):

    def __call__(self, img):
        return img.rotate(90 * random.randint(0, 4))


class RandomScale(object):

    def __call__(self, img):
        shape = img.size
        ratio = min(shape)/CROP_SIZE
        scale = random.uniform(ratio, 1)
        return img.resize((int(shape[0]*scale), int(shape[1]*scale)), Image.BICUBIC)




def LR_transform(crop_size):
    return Compose([
        Scale(crop_size//8),
        ToTensor(),
    ])


def HR_2_transform(crop_size):
    return Compose([
        Scale(crop_size//4),
        ToTensor(),
    ])


def HR_4_transform(crop_size):
    return Compose([
        Scale(crop_size//2),
        ToTensor(),
    ])


def HR_8_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        RandomScale(),
        #RandomRotate(),
        RandomHorizontalFlip(),
    ])


def get_training_set(train_dir=None):
    return DatasetFromFolder(train_dir,
                             LR_transform=LR_transform(CROP_SIZE),
                             HR_8_transform=HR_8_transform(CROP_SIZE))


