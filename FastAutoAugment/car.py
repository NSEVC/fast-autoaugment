# -*- coding: utf-8 -*-
# @Author  : chenlijuan
# @File    : car.py
# @Time    : 2020/2/12 下午9:06
# @Desc    :

from __future__ import print_function
import os
import shutil
import torch

import torchvision
from torchvision.datasets.utils import check_integrity, download_url


class Car(torchvision.datasets.ImageFolder):
    def __init__(self, root, split='train', **kwargs):
        root = self.root = os.path.expanduser(root)   # 把root中包含的"~"和"~user"转换成用户目录
        self.split = self._verify_split(split)

        listfile = os.path.join(root, 'train_cls.txt')    # clss/filename index
        if split == 'train' and os.path.exists(listfile):
            torchvision.datasets.VisionDataset.__init__(self, root, **kwargs)
            with open(listfile, 'r') as f:
                datalist = [line.strip().split(' ')[0] for line in f.readlines() if line.strip()]

            classes = list(set([line.split('/')[0] for line in datalist]))
            classes.sort()
            class_to_idx = {classes[i]: i for i in range(len(classes))}

            samples = [
                (os.path.join(self.split_folder, line), class_to_idx[line.split('/')[0]])  # self.split_folder=root+"train"+dir+filename
                for line in datalist
            ]

            self.loader = torchvision.datasets.folder.default_loader
            self.extensions = torchvision.datasets.folder.IMG_EXTENSIONS

            self.classes = classes
            self.class_to_idx = class_to_idx
            self.samples = samples
            self.targets = [s[1] for s in samples]

            self.imgs = self.samples
        else:
            super(Car, self).__init__(self.split_folder, **kwargs)

        self.root = root

        idcs = [idx for _, idx in self.imgs]
        self.class_to_idx = {cls: idx
                             for clss, idx in zip(self.classes, idcs)
                             for cls in clss}

    def _verify_split(self, split):
        if split not in self.valid_splits:
            msg = "Unknown split {} .".format(split)
            msg += "Valid splits are {{}}.".format(", ".join(self.valid_splits))
            raise ValueError(msg)
        return split

    @property
    def valid_splits(self):
        return 'train', 'val'

    @property
    def split_folder(self):
        return os.path.join(self.root, self.split)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)


def _splitexts(root):
    exts = []
    ext = '.'
    while ext:
        root, ext = os.path.splitext(root)
        exts.append(ext)
    return root, ''.join(reversed(exts))
