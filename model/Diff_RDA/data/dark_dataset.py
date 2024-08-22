import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import matplotlib.pyplot as plt


class Dark_Dataset(Dataset):
    def __init__(self, dataroot, resolution=256, split='train', data_len=20):
        self.resolution = resolution
        self.data_len = data_len
        self.split = split

        self.low_path = Util.get_paths_from_images('{}/low'.format(dataroot))
        self.high_path = Util.get_paths_from_images('{}/high'.format(dataroot))

        self.dataset_len = len(self.high_path)

        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_low = Image.open(self.low_path[index]).convert("RGB")
        img_high = Image.open(self.high_path[index]).convert("RGB")

        [img_LOW, img_HIGH] = Util.transform_augment([img_low, img_high], split=self.split, min_max=(-1, 1))

        if self.split == "val":
            path = str(self.low_path[index]).replace("\\","/")
            name = str(path.split("/")[-1].split(".png")[0])
            return {'high': img_HIGH, 'low': img_LOW, 'Index': index}, name

        return {'high': img_HIGH, 'low': img_LOW, 'Index': index}