import random
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import torch.nn as nn
import torchvision.transforms as transforms


class get_aug(nn.Module):
    
    def __init__(self, cfgs, mode, **kwargs):
        super(get_aug, self).__init__()
        self.cfgs = cfgs
        self.mode = mode
        if cfgs.dataset.modality == '2D':
            if mode == 'train':
                self.trans = iaa.Sequential([ 
                    iaa.Affine(scale={"x": (cfgs.data_aug.affine.scale[0], cfgs.data_aug.affine.scale[1]),\
                    "y": (cfgs.data_aug.affine.scale[0], cfgs.data_aug.affine.scale[1])},\
                    translate_percent={"x": (-cfgs.data_aug.affine.translate[0], cfgs.data_aug.affine.translate[0]),\
                    "y": (-cfgs.data_aug.affine.translate[0], cfgs.data_aug.affine.translate[0])},\
                    rotate=(cfgs.data_aug.affine.rotate[0], cfgs.data_aug.affine.rotate[1]), \
                    shear=(cfgs.data_aug.affine.shear[0], cfgs.data_aug.affine.shear[1])),
                    iaa.flip.HorizontalFlip(cfgs.data_aug.flip[0]),
                    iaa.flip.VerticalFlip(cfgs.data_aug.flip[1]),
                    iaa.AddToHueAndSaturation((-50, 50), per_channel=True)
                    ], random_order=False)
            else:
                pass

        elif cfgs.dataset.modality == '3D':
            if 'augmentor' in kwargs:
                self.trans = kwargs['augmentor']
            else:
                raise NotImplementedError

    def forward(self, img, mask):
        if self.cfgs.dataset.modality == '2D':
            if self.mode == 'train':
                mask = SegmentationMapsOnImage(mask, shape=img.shape)
                img_aug, mask_aug = self.trans(image=img, segmentation_maps=mask)
                mask_aug = mask_aug.get_arr()
            else:
                img_aug = img
                mask_aug = mask

            img_aug = ((img_aug - self.cfgs.dataset.mean)/self.cfgs.dataset.std)

        elif self.cfgs.dataset.modality == '3D':
            img_aug, mask_aug = self.trans(img, mask)

        return img_aug, mask_aug

