import argparse
import os
import pickle
import yaml
import shutil
import time
import math
import glob
import re
import numpy as np
import datasets
import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from utils.LARC import LARC
from addict import Dict
from utils import dist_utils, helper_utils, log_utils, loss_utils, metric_utils, schd_utils, save_vis_utils
from thop import profile
import PIL.Image as Image


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Amoeba')
    parser.add_argument('--config', default='configs/tem.yaml',
                        type=str, help='path to config file')
    parser.add_argument('--save_path', default='./runs/', type=str, help='path to save checkpoint')
    parser.add_argument('--resume_from', default=None, type=str, help='path to checkpoint')
    parser.add_argument("--port", default='23333') 
    parser.add_argument("--local_rank", default=None)   
    return parser.parse_args()


def main(args):
    cfgs = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfgs = Dict(cfgs)

    dist_utils.init_distributed_mode(args)

    dist_utils.init_seeds(seed=cfgs.seed, cuda_deterministic=False)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.rank == 0:
        helper_utils.copy_file(args)
        args.logger, args.val_stats = helper_utils.init_log(args, cfgs)

    if args.rank == 0:
        args.logger.info('Loading data ...')

    test_dataset = datasets.__dict__[cfgs.dataset.mode](cfgs, mode='test')
    test_sampler = dist_utils.SequentialDistributedSampler(test_dataset, batch_size=1)                         
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             sampler=test_sampler,
                                             batch_size=1,
                                             num_workers=cfgs.num_workers,
                                             drop_last=False,
                                             pin_memory=True)  

    cfgs.test_len = len(test_dataset)

    model = models.__dict__[cfgs.model.mode](args, cfgs)

    if args.rank == 0:
        args.logger.info("Building model done.")

    if cfgs.syncbn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.to(args.device)
    if args.world_size > 0:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    state_dict = torch.load('....pth')['state_dict']
    model.load_state_dict(state_dict)
    result_dict_all, result_dict = val(args, cfgs, val_loader, model, criterion, 12)



def val(args, cfgs, val_loader, model, criterion, epoch):

    model.eval()
    eval_result = []
    evaluator = metric_utils.Evaluator(cfgs)

    with torch.no_grad():
        for it, (img, resize_mask, gt_mask, img_path, mask_path) in enumerate(val_loader):
            if len(img.size()) == 4:
                img = img.float().to(args.device).permute(0, 3, 1, 2).contiguous()  
                gt_mask = gt_mask.long().to(args.device) 
            elif len(img.size()) == 5:
                img = img.float().to(args.device)  
                gt_mask = gt_mask.long().to(args.device)  
            else:
                raise NotImplementedError

            pred = model(img, resize_mask.unsqueeze(1))

            if cfgs.val.resize_option == 'resize_first':
                pred_mask = helper_utils.resize_before(pred, gt_mask.size(), cfgs)
            elif cfgs.val.resize_option == 'resize_after':
                pred_mask = helper_utils.resize_after(pred, gt_mask.size(), cfgs)
            else:
                raise NotImplementedError

            evaluator.add_batch(gt_mask, pred_mask.unsqueeze(1))

        evaluator.distributed_concat()
        result_dict_all, result_dict  = evaluator.get_metric()

    return result_dict_all, result_dict


if __name__ == '__main__':
    args = parse_args()
    main(args)
