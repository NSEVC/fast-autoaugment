# -*- coding: utf-8 -*-
# @Author  : chenlijuan
# @File    : car.py
# @Time    : 2020/2/12 下午9:06
# @Desc    :

from __future__ import print_function
import os
import shutil
import torch
from PIL import Image

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
                datalist = [" ".join(line.strip().split(' ')[:-1]) for line in f.readlines() if line.strip()]

            classes = list(set([line.split('/')[0] for line in datalist]))
            classes.sort()
            class_to_idx = {classes[i]: i for i in range(len(classes))}

            samples = [
                (os.path.join(self.split_folder, line), class_to_idx[line.split('/')[0]])  # self.split_folder=root+"train"+dir+filename
                for line in datalist
            ]

            # self.loader = torchvision.datasets.folder.default_loader
            self.loader = self.loader
            self.extensions = torchvision.datasets.folder.IMG_EXTENSIONS

            self.classes = classes
            self.class_to_idx = class_to_idx
            self.samples = samples
            self.targets = [s[1] for s in samples]

            self.imgs = self.samples
        else:
            super(Car, self).__init__(self.split_folder, loader=self.loader, **kwargs)

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

    def pad_image(self, image, target_size):
        iw, ih = image.size  # 原始图像的尺寸
        w, h = target_size  # 目标图像的尺寸
        scale = min(w / iw, h / ih)  # 转换的最小比例

        # 保证长或宽，至少一个符合目标图像的尺寸
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)  # 缩小图像
        image.show()
        new_image = Image.new('RGB', target_size, (128, 128, 128))  # 生成灰色图像
        # // 为整数除法，计算图像的位置
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 将图像填充为中间图像，两侧为灰色的样式

        return new_image

    def loader(self, path):
        image_size = 380
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
            return self.pad_image(img, (image_size, image_size))

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
