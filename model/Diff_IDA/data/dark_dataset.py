import numpy as np
from PIL import Image
from PIL import ImageEnhance
from torch.utils.data import Dataset
import random
import data.util as Util
import matplotlib.pyplot as plt
import torch


class Dark_Dataset(Dataset):
    def __init__(self, dataroot, resolution=256, split='train', data_len=20):
        self.resolution = resolution
        self.data_len = data_len
        self.split = split

        self.low_path = Util.get_paths_from_images('{}/low'.format(dataroot))
        self.high_path = Util.get_paths_from_images('{}/high'.format(dataroot))
        self.R_ref_path = Util.get_paths_from_images('{}/R'.format(dataroot))
        self.gt_path = Util.get_paths_from_images('{}/gt'.format(dataroot))

        self.dataset_len = len(self.high_path)

        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_low = Image.open(self.low_path[index]).convert("L")
        img_high = Image.open(self.high_path[index]).convert("L")
        img_R_ref = Image.open(self.R_ref_path[index]).convert("RGB")
        img_gt_ref = Image.open(self.gt_path[index]).convert("RGB")

        img_low = img_low.convert("RGB")
        img_high = img_high.convert("RGB")

        [img_LOW, img_HIGH, img_R, img_gt] = Util.transform_augment([img_low, img_high, img_R_ref, img_gt_ref], split=self.split, min_max=(-1, 1))
        img_LOW = torch.mean(img_LOW, dim=0, keepdim=True)
        img_HIGH = torch.mean(img_HIGH, dim=0, keepdim=True)

        if self.split == "val":
            img_R = Util.transform_full(img_R_ref, min_max=(-1, 1))
            img_gt = Util.transform_full(img_gt_ref, min_max=(-1, 1))
            path = str(self.low_path[index]).replace("\\", "/")
            name = str(path.split("/")[-1].split(".png")[0])
            return {'high': img_HIGH, 'low': img_LOW, 'R': img_R, 'gt': img_gt, 'Index': index}, name

        return {'high': img_HIGH, 'low': img_LOW, 'R':img_R, 'gt': img_gt, 'Index': index}