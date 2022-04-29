import os
import glob
import copy
import numpy as np
import PIL.Image as Image
import torch
from .mclient_reader import MClientReader
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from .data_aug import get_aug

try:
    import sys
    import mc
    sys.path.append('/mnt/lustre/share/pymc/py3/')
except:
    print('memcache not exist!')


__all__ = ['ISIC2016']


class ISIC2016(Dataset):
    def __init__(self, cfgs, mode='train'):
        if mode == 'train':
            sub_dir = 'train'
        else:
            sub_dir = 'test'

        self.cfgs = cfgs
        self.mode = mode
        self.imgs = make_dataset(cfgs, cfgs.dataset.root_path + '/' + cfgs.dataset.mode + '/' + sub_dir)
        self.augmentor = get_aug(cfgs, mode)

        if cfgs.mc:
            self.image_reader = MClientReader()
        else:
            self.image_reader = None

    def __getitem__(self, index):
        if self.cfgs.dataset.preload:
            img, mask = self.imgs[index]
        else:
            img_path, mask_path = self.imgs[index]
            if self.image_reader:
                img = self.image_reader.open(img_path)
                mask = self.image_reader.open(mask_path, binary=True)
            else:
                img = Image.open(img_path).convert('RGB')
                mask = Image.open(mask_path).convert('L')

        img = img.resize((self.cfgs.dataset.input_size[0], self.cfgs.dataset.input_size[1]), resample=3)
        if self.mode == 'train':
            mask = mask.resize((self.cfgs.dataset.input_size[0], self.cfgs.dataset.input_size[1]), resample=1)
        img, mask = np.array(img), np.array(mask)
        img, mask = img.copy(), mask.copy()
        mask[mask > 0] = 1
        if self.cfgs.dataset.aug:
            img, mask = self.augmentor(img, mask)
        mask = np.expand_dims(mask, 0)

        return img, mask

    def __len__(self):
        return len(self.imgs)


def make_dataset(cfgs, root):
    input_collection = []
    img_types = ['.jpg', '.jpeg', '.png', '.JPG']
    img_paths, mask_paths = [], []
    for img_type in img_types:
        img_paths.extend(sorted(glob.glob(root + '/img/*' + img_type)))
        mask_paths.extend(sorted(glob.glob(root + '/mask_jpg/*' + img_type)))

    for img_path in img_paths:
        pure_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = [path for path in mask_paths if pure_name in path][0]
        input_collection.append((img_path, mask_path))

    if cfgs.dataset.preload:
        input_collection = [(Image.open(item[0]).convert('RGB'), Image.open(item[1]).convert('L')) for item in input_collection]
    else:
        pass

    return input_collection


