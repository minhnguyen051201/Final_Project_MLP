import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from random import random
import os
from os.path import join, dirname, exists
from data.dataset_utils import *

class RotateDataset(data.Dataset):
    def __init__(self, name, split='train', val_size=0, rot_classes=3,
            img_transformer=None, bias_whole_image=None, root_dir=None):
        # Prefer directory-based listing if a matching root/name folder exists
        prefer_dir = False
        scan_root = root_dir if root_dir else join(dirname(__file__), '..', 'datasets')
        if os.path.isdir(os.path.join(scan_root, name)):
            prefer_dir = True
        if split == 'train':
            if not prefer_dir:
                try:
                    names, _, labels, _ = get_split_dataset_info(join(dirname(__file__), 'txt_lists', '%s_train.txt' % name), val_size)
                except Exception:
                    prefer_dir = True
            if prefer_dir:
                names, labels = build_dataset_from_dir(scan_root, name)
                names, _, labels, _ = get_random_subset(names, labels, val_size)
        elif split =='val':
            if not prefer_dir:
                try:
                    _, names, _, labels = get_split_dataset_info(join(dirname(__file__), 'txt_lists', '%s_train.txt' % name), val_size)
                except Exception:
                    prefer_dir = True
            if prefer_dir:
                names, labels = build_dataset_from_dir(scan_root, name)
                _, names, _, labels = get_random_subset(names, labels, val_size)
        elif split == 'test':
            if not prefer_dir:
                try:
                    names, labels = get_dataset_info(join(dirname(__file__), 'txt_lists', '%s_test.txt' % name))
                except Exception:
                    prefer_dir = True
            if prefer_dir:
                names, labels = build_dataset_from_dir(scan_root, name)

        # Allow overriding the dataset root via config
        default_root = join(dirname(__file__), '..', 'datasets')
        self.data_path = root_dir if root_dir else default_root
        # Auto-correct root if provided path doesn't match txt list prefix
        try:
            if len(names) > 0 and not exists(join(self.data_path, names[0])):
                parent = dirname(self.data_path)
                if exists(join(parent, names[0])):
                    self.data_path = parent
                elif exists(join(default_root, names[0])):
                    self.data_path = default_root
        except Exception:
            pass
        self.names = names
        self.labels = labels
        self.rot_classes = rot_classes

        self.N = len(self.names)
        self.bias_whole_image = bias_whole_image
        self._image_transformer = img_transformer

    def rotate_all(self, img):
        """Rotate for all angles"""
        img_rts = []
        for lb in range(self.rot_classes + 1):
            img_rt = self.rotate(img, rot=lb * 90)
            img_rts.append(img_rt)

        return img_rts

    def rotate(self, img, rot):
        if rot == 0:
            img_rt = img
        elif rot == 90:
            img_rt = img.transpose(Image.ROTATE_90)
        elif rot == 180:
            img_rt = img.transpose(Image.ROTATE_180)
        elif rot == 270:
            img_rt = img.transpose(Image.ROTATE_270)
        else:
            raise ValueError('Rotation angles should be in [0, 90, 180, 270]')
        return img_rt

    def get_image(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        return img

    def __getitem__(self, index):
        img = self.get_image(index)
        rot_imgs = self.rotate_all(img)

        order = np.random.randint(self.rot_classes + 1)  # added 1 for class 0: unrotated
        if self.bias_whole_image:
            if self.bias_whole_image > random():
                order = 0

        data = rot_imgs[order]
        data = self._image_transformer(data)
        sample = {'images': data,
                'aux_labels': int(order),
                'class_labels': int(self.labels[index])}
        return sample

    def __len__(self):
        return len(self.names)

class RotateTestDataset(RotateDataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')

        sample = {'images': self._image_transformer(img),
                'aux_labels': 0,
                'class_labels': int(self.labels[index])}
        return sample
