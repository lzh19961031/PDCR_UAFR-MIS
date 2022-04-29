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
from utils import dist_utils, helper_utils, log_utils, loss_utils, metric_utils, schd_utils


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
    train_dataset = datasets.__dict__[cfgs.dataset.mode](cfgs, mode='train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               sampler=train_sampler,
                                               batch_size=cfgs.batch_size,
                                               num_workers=cfgs.num_workers,
                                               drop_last=True,
                                               pin_memory=True)

    test_dataset = datasets.__dict__[cfgs.dataset.mode](cfgs, mode='test')
    test_sampler = dist_utils.SequentialDistributedSampler(val_dataset, batch_size=1)                         
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             sampler=test_sampler,
                                             batch_size=1,
                                             num_workers=cfgs.num_workers,
                                             drop_last=False,
                                             pin_memory=True)  

    cfgs.train_len = len(train_dataset)
    cfgs.test_len = len(test_dataset)

    model = models.__dict__[cfgs.model.mode](args, cfgs)

    if args.rank == 0:
        args.logger.info("Building model done.")

    model.to(args.device)
    if cfgs.syncbn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.to(args.device)

    if cfgs.optm.mode == 'sgd':
        base_lr = cfgs.optm.sgd.base_lr
        final_lr = cfgs.optm.sgd.final_lr
        wd = cfgs.optm.sgd.wd
        momentum = cfgs.optm.sgd.momentum
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=base_lr,
            momentum=momentum,
            weight_decay=wd)
    elif cfgs.optm.mode == 'adam':
        base_lr = cfgs.optm.adam.base_lr
        final_lr = cfgs.optm.adam.final_lr
        wd = cfgs.optm.adam.wd
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=base_lr,
            weight_decay=wd)
    else:
        raise NotImplementedError

    if cfgs.optm.use_larc:
        trust_coefficient = cfgs.optm.larc.trust_coefficient
        clip = cfgs.optm.larc.clip
        optimizer = LARC(optimizer=optimizer, trust_coefficient=trust_coefficient, clip=clip)

    if args.rank == 0:
        args.logger.info("Building optimizer done.")

    lr_scheduler = schd_utils.get_scheduler(cfgs, train_loader)

    criterion = loss_utils.get_criterion(args, cfgs)

    if args.resume_from is not None:
        if args.rank == 0:
            args.logger.info('Resume from an existing checkpoint!')
        checkpoint = torch.load(args.resume_from, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    rank_metric = {}
    for metric_name in cfgs.test.rank_metric:
        rank_metric[metric_name] = 0

    for epoch in range(start_epoch, cfgs.epochs):
        if args.rank == 0:
            args.logger.info("============ Starting epoch %i ... ============" % epoch)

        train_loader.sampler.set_epoch(epoch)

        train(args, cfgs, train_loader, model, criterion, optimizer, lr_scheduler, epoch)

        if args.rank == 0 and epoch % cfgs.save_checkpoint == 0:
            path = os.path.join(args.checkpoints_path, 'epoch_' + str(epoch) + '.pth')
            save_dict = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(save_dict, path)  


def train(args, cfgs, train_loader, model, criterion, optimizer, lr_scheduler, epoch):
    batch_time = helper_utils.AverageMeter()
    data_time = helper_utils.AverageMeter()
    losses = helper_utils.AverageMeter()

    model.train()

    end = time.time()
    for it, (img, gt_mask) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if len(img.size()) == 4:
            img = img.float().to(args.device).permute(0, 3, 1, 2).contiguous()  
            gt_mask = gt_mask.long().to(args.device)  
        elif len(img.size()) == 5:
            img = img.float().to(args.device)  
            gt_mask = gt_mask.long().to(args.device)  
        else:
            raise NotImplementedError

        pred = model(img, gt_mask)

        loss = criterion.forward(pred, gt_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iteration = epoch * len(train_loader) + it
        lr_scheduler.it_step(optimizer, iteration)

        losses.update(loss.item(), gt_mask[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if args.rank == 0 and (it % 50 == 0 or it % (len(train_loader)-1) == 0):
            args.logger.info(
                "Epoch: [{0}][{1}]\t"
                "Max_GPUmemory: {2}\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.6f})\t"
                "Lr: {lr:.6f}".format(
                    epoch,
                    it,
                    torch.cuda.max_memory_allocated(), 
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=optimizer.param_groups[0]["lr"]))


def test(args, cfgs, test_loader, model, criterion, epoch):

    model.eval()
    eval_result = []
    evaluator = metric_utils.Evaluator(cfgs)

    with torch.no_grad():
        for it, (img, gt_mask) in enumerate(test_loader):

            if len(img.size()) == 4:
                img = img.float().to(args.device).permute(0, 3, 1, 2).contiguous()  
                gt_mask = gt_mask.long().to(args.device)  
            elif len(img.size()) == 5:
                img = img.float().to(args.device)
                gt_mask = gt_mask.long().to(args.device) 
            else:
                raise NotImplementedError

            pred = model(img, gt_mask)

            if cfgs.test.resize_option == 'resize_first':
                pred_mask = helper_utils.resize_before(pred, gt_mask.size(), cfgs)
            elif cfgs.test.resize_option == 'resize_after':
                pred_mask = helper_utils.resize_after(pred, gt_mask.size(), cfgs)
            else:
                raise NotImplementedError

            evaluator.add_batch(gt_mask, pred_mask.unsqueeze(1))

        evaluator.distributed_concat()

        result_dict = evaluator.get_metric()

    return result_dict


if __name__ == '__main__':
    args = parse_args()
    main(args)
