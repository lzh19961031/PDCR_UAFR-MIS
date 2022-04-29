import os
import glob
import random
import numpy as np
import PIL.Image as Image
import SimpleITK as sitk
import torch
from .mclient_reader import MClientReader
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from scipy.ndimage import affine_transform, zoom, rotate
from .data_aug import get_aug
try:
    import sys
    import mc
    sys.path.append('/mnt/lustre/share/pymc/py3/')
except:
    print('memcache not exist!')


__all__ = ['Plaques']


class Plaques(Dataset):
    def __init__(self, cfgs, mode='train'):
        if mode == 'train':
            sub_dir = 'train'
        else:
            sub_dir = 'val'

        self.cfgs = cfgs
        self.mode = mode
        data_paths = get_data_path(cfgs.dataset.root_path + '/' + sub_dir)
        self.index_list, self.env_list, self.labelweights = prepare_data(data_paths, self.cfgs.model.num_classes)
        self.cfgs.class_weight = self.labelweights 
        self.augmentor = get_aug(cfgs, mode, augmentor=augmentation_3d)

    def __getitem__(self, index):

        pt_idx, env_idx = self.index_list[index]
        cur_env_dict = self.env_list[env_idx]
        mpr, mask = cur_env_dict['img'].astype(np.float64), cur_env_dict['mask'].astype(np.uint8)
        assert mpr.shape == mask.shape, print('shape not match!')

        coordinate = [pt_idx, mpr.shape[1]/2, mpr.shape[2]/2]
        if self.cfgs.dataset.aug:
            # random shift
            shift_radius = 0.5
            if random.random() > 0.5:
                mean = coordinate
                cov = [[shift_radius, 0, 0], [0, shift_radius, 0], [0, 0, shift_radius]] 
                x, y, z = np.random.multivariate_normal(mean, cov, 1).T
                coordinate = [x[0], y[0], z[0]]

        # crop patch
        mpr_patch = roi_crop_pad(mpr, coordinate, self.cfgs.dataset.input_size)
        mask_patch = roi_crop_pad(mask, coordinate, self.cfgs.dataset.input_size)

        # data augmentation
        if self.cfgs.dataset.aug:
            mpr_patch, mask_patch = self.augmentor(mpr_patch, mask_patch)

        mask_patch = mask_patch.astype(np.uint8)
        # save_image(mpr_patch, '/home/SENSETIME/lizhuowei/Desktop/plaque_tem/' + str(idx) + '-mpr.nii.gz')
        # save_image(mask_patch, '/home/SENSETIME/lizhuowei/Desktop/plaque_tem/' + str(idx) + '-mask.nii.gz')

        # normalization
        if np.amax(mpr_patch) > 1:
            mpr_patch = normalize(mpr_patch)

        # make 5D input
        mpr_patch = np.expand_dims(mpr_patch, axis=0)
        mask_patch = np.expand_dims(mask_patch, axis=0)

        return mpr_patch, mask_patch

    def __len__(self):
        return len(self.index_list)


def get_data_path(root_path):
    data_paths = []
    target_paths = sorted(glob.glob(root_path + '/**/**/'))
    for path in target_paths:
        annotated = False
        for dir_path, dir_names, file_names in os.walk(path):
            for filename in file_names:
                if 'mask' in filename:
                    annotated = True
                    break
        if annotated:
            data_paths.append(path)
    random.shuffle(data_paths)
    return data_paths


def save_image(image, path, origin=None, spacing=None, direction=None):

    # image = np.transpose(image, (2,1,0))
    itk_image = sitk.GetImageFromArray(image)
    
    if spacing is not None:
        itk_image.SetSpacing(spacing)
    else:
        itk_image.SetSpacing([1,1,1])
        
    if origin is not None:
        itk_image.SetOrigin(origin) # default: 0, 0, 0
        
    if direction is not None:
        itk_image.SetDirection(direction) # default: 100,010,001
    
    sitk.WriteImage(itk_image, path)


def roi_crop_pad(image, center, bbox_size):

    bbminx = int(center[0] - bbox_size[0] // 2)
    bbminy = int(center[1] - bbox_size[1] // 2)
    bbminz = int(center[2] - bbox_size[2] // 2)

    bbmaxx = bbminx + bbox_size[0]
    bbmaxy = bbminy + bbox_size[1]
    bbmaxz = bbminz + bbox_size[2]

    pad_xl, pad_yl, pad_zl = 0, 0, 0
    pad_xr, pad_yr, pad_zr = 0, 0, 0

    if bbminx < 0:
        pad_xl = -bbminx
        bbminx = 0

    if bbminy < 0:
        pad_yl = -bbminy
        bbminy = 0

    if bbminz < 0:
        pad_zl = -bbminz
        bbminz = 0

    if bbmaxx > image.shape[0]:
        pad_xr = bbmaxx - image.shape[0]
        bbmaxx = image.shape[0]

    if bbmaxy > image.shape[1]:
        pad_yr = bbmaxy - image.shape[1]
        bbmaxy = image.shape[1]

    if bbmaxz > image.shape[2]:
        pad_zr = bbmaxz - image.shape[2]
        bbmaxz = image.shape[2]

    image = image[bbminx:bbmaxx, bbminy:bbmaxy, bbminz:bbmaxz]
    image = np.pad(image, ((pad_xl, pad_xr), (pad_yl, pad_yr), (pad_zl, pad_zr)), mode='reflect')

    return image


def prepare_data(data_paths, n_classes):

    env_list = []
    env_count = 0
    labelweights = np.zeros(n_classes)

    for file_path in data_paths:

        try:
            if os.path.exists(file_path + '/mpr_100.nii.gz'):
                mpr_path = file_path + '/mpr_100.nii.gz'
            else:
                mpr_path = file_path + '/mpr.nii.gz' 

            if os.path.exists(file_path + '/mask_refine_checked.nii.gz'):
                mask_path = file_path + '/mask_refine_checked.nii.gz'
            elif os.path.exists(file_path + '/mask_refine.nii.gz'):
                mask_path = file_path + '/mask_refine.nii.gz'
            else:
                mask_path = file_path + '/mask.nii.gz' 

            mpr_itk = sitk.ReadImage(mpr_path)
            mask_itk = sitk.ReadImage(mask_path)
            mpr_vol = sitk.GetArrayFromImage(mpr_itk) # shape: (length, x, y)
            mask_vol = sitk.GetArrayFromImage(mask_itk) # shape: (length, x, y)
            assert mpr_vol.shape == mask_vol.shape, print('Wrong shape')

            # remove anchor voxels
            mask_vol[mask_vol>5] = 0

            if n_classes == 4:
                mask_vol[mask_vol==4]=0
            elif n_classes == 3:
                mask_vol[mask_vol==4]=2
            else:
                pass

            unique, counts = np.unique(mask_vol, return_counts=True)
            labelweights[unique.astype(int)] += counts

            cur_id_list = []
            for i in range(mask_vol.shape[0]):
                if mask_vol[i, int((mask_vol.shape[1]-1)/2), int((mask_vol.shape[2]-1)/2)] != 0: # there is mask in center
                    cur_id_list.append(i)
                else:
                    if len(cur_id_list) == 0:
                        pass
                    else:
                        sub_mpr = mpr_vol[cur_id_list[0]:cur_id_list[-1]+1, :, :]
                        sub_mask = mask_vol[cur_id_list[0]:cur_id_list[-1]+1, :, :]
                        sub_env_dict = {'img': sub_mpr, 'mask': sub_mask.astype(np.uint8)}
                        env_list.append(sub_env_dict)
                        cur_id_list = []
        except:
            print('dirty data')

    # use weight to balance data distribution
    labelweights = labelweights / (np.sum(labelweights) + 1e-8)
    labelweights = np.power(np.amax(labelweights) / (labelweights + 1e-8), 1 / 3.0)
    
    index_list = []
    for env_idx, item in enumerate(env_list):
        # save_image(item['img'], '/home/SENSETIME/lizhuowei/Desktop/plaque_tem/' + str(idx) + '-mpr.nii.gz')
        # save_image(item['mask'], '/home/SENSETIME/lizhuowei/Desktop/plaque_tem/' + str(idx) + '-mask.nii.gz')
        for i in range(item['img'].shape[0]):
            index_list.append((i, env_idx))

    return index_list, env_list, labelweights


def augmentation_3d(img, mask):
    # rotation
    if random.random() >= 0.5:
        degree_range = 180
        degree = random.randrange(-degree_range, degree_range)
        img, mask = rotation(img, mask, degree)
        
    # intensity shift
    if random.random() >= 0.5:
        degree = np.random.uniform(low=-0.1, high=0.1)
        img = adjust_HU(img, degree)
    return img, mask


def adjust_HU(img, diff):
    img = img + np.amax(img) * diff
    img = img * (1 + diff)
    return img


def rotation(img, mask, degree):
    img = rotate(img, degree, (1,2), reshape=False, mode='nearest', order=1)
    mask = rotate(mask, degree, (1,2), reshape=False, mode='nearest', order=0)
    return img, mask


def normalize(img):
    img = np.clip(img, -360, 840)
    img = (img + 360) / 1200
    return img
