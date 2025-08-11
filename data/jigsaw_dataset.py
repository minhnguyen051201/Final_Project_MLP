import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from random import sample, random
import os
from os.path import join, dirname, exists
from data.dataset_utils import *

class JigsawDataset(data.Dataset):
    def __init__(self, name, split='train', val_size=0, jig_classes=100,
            img_transformer=None, tile_transformer=None, patches=False, bias_whole_image=None, root_dir=None):
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

        self.N = len(self.names)
        self.permutations = self.__retrieve_permutations(jig_classes)
        self.grid_size = 3
        self.bias_whole_image = bias_whole_image
        if patches:
            self.patch_size = 64
        self._image_transformer = img_transformer
        self._augment_tile = tile_transformer
        if patches:
            self.returnFunc = lambda x: x
        else:
            def make_grid(x):
                return torchvision.utils.make_grid(x, self.grid_size, padding=0)
            self.returnFunc = make_grid

    def get_tile(self, img, n):
        w = float(img.size[0]) / self.grid_size
        y = int(n / self.grid_size)
        x = n % self.grid_size
        tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
        tile = self._augment_tile(tile)
        return tile

    def get_image(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img)

    def __getitem__(self, index):
        img = self.get_image(index)
        n_grids = self.grid_size ** 2
        tiles = [None] * n_grids
        for n in range(n_grids):
            tiles[n] = self.get_tile(img, n)

        order = np.random.randint(len(self.permutations) + 1)  # added 1 for class 0: unsorted
        if self.bias_whole_image:
            if self.bias_whole_image > random():
                order = 0
        if order == 0:
            data = tiles
        else:
            data = [tiles[self.permutations[order - 1][t]] for t in range(n_grids)]

        data = torch.stack(data, 0)
        sample = {'images': self.returnFunc(data),
                'aux_labels': int(order),
                'class_labels': int(self.labels[index])}
        return sample

    def __len__(self):
        return len(self.names)

    def __retrieve_permutations(self, classes):
        all_perm = np.load(join(dirname(__file__), 'permutations_%d.npy' % (classes)))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm

class JigsawTestDataset(JigsawDataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')

        sample = {'images': self._image_transformer(img),
                'aux_labels': 0,
                'class_labels': int(self.labels[index])}
        return sample
