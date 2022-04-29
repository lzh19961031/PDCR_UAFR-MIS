# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from logging import getLogger
import pickle
import os
import csv
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from .log_utils import create_logger, pd_stats


def copy_file(args):
    save_path = args.save_path
    checkpoints_path = os.path.join(save_path, 'checkpoints')
    args.checkpoints_path = checkpoints_path

    # create save folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # create checkpoint folder
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    # copy relevant files
    if not os.path.exists(save_path + '/criterions'):
        shutil.copytree('./criterions', save_path + '/criterions')
    if not os.path.exists(save_path + '/datasets'):
        shutil.copytree('./datasets', save_path + '/datasets')
    if not os.path.exists(save_path + '/models'):
        shutil.copytree('./models', save_path + '/models')
    if not os.path.exists(save_path + '/utils'):
        shutil.copytree('./utils', save_path + '/utils')
    if not os.path.exists(save_path + '/' + os.path.split(args.config)[1]):
        shutil.copyfile(args.config, save_path + '/' + os.path.split(args.config)[1])


def init_log(args, cfgs):
    # create a panda object to log loss and acc
    column_field = ['epoch'] + cfgs.val.metric_used
    val_stats = pd_stats(os.path.join(args.checkpoints_path, "stats.csv"), column_field)

    # create a logger
    logger = create_logger(os.path.join(args.checkpoints_path, "train.log"), args.rank)
    logger.info("============ Initialized logger ============")
    logger.info("The experiment will be stored in %s\n" % args.checkpoints_path)
    logger.info("")

    return logger, val_stats


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def rename_key(key):
    if not 'module' in key:
        return key
    return ''.join(key.split('module.'))


def resize_before(pred, gt_size, cfgs=None):
    """ 
    pred_mask size: (N, C, d1, d2 .... dn)    
    gt_size: (N, d1, d2, ... dn)
    """
    if type(pred) == list or type(pred) == tuple:
        pred_mask = pred[0]
    else:
        pred_mask = pred

    target_size = gt_size[2:]
    pred_mask = F.interpolate(pred_mask, target_size, mode=cfgs.val.resize_interpolation)
    pred_mask = torch.argmax(pred_mask, dim=1)    
    return pred_mask


def resize_after(pred, gt_size, cfgs=None):

    """ 
    pred_mask size: (N, C, d1, d2 .... dn)    
    gt_size: (N, d1, d2, ... dn)
    """
    if type(pred) == list or type(pred) == tuple:
        pred_mask = pred[0]
    else:
        pred_mask = pred
    target_size = gt_size[2:]
    pred_mask = torch.argmax(pred_mask, dim=1)
    pred_mask = pred_mask.float().unsqueeze(1)
    pred_mask = F.interpolate(pred_mask, target_size, mode='nearest')
    pred_mask = pred_mask.long().squeeze(1)  
    return pred_mask