from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, images_low_path: list, images_high_path: list, transform=None):
        self.images_low_path = images_low_path
        self.images_high_path = images_high_path
        self.transform = transform

    def __len__(self):
        return len(self.images_low_path)

    def __getitem__(self, item):
        img = Image.open(self.images_low_path[item])
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_low_path[item]))
        img_ref = Image.open(self.images_high_path[item])
        if img_ref.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_high_path[item]))

        if self.transform is not None:
            img, img_ref = self.transform(img, img_ref)

        return img, img_ref

    @staticmethod
    def collate_fn(batch):
        images, images_ref = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        images_ref = torch.stack(images_ref, dim=0)
        return images, images_ref
