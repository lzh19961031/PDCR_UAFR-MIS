import torch
from random import sample
import random
import torch.nn as nn
import yaml
from addict import Dict
from itertools import combinations, permutations
import numpy as np

class CL_Loss(torch.nn.Module):

    def __init__(self, args, cfgs):
        super(CL_Loss, self).__init__()
        self.cfgs = cfgs
        self.batch_size = cfgs.batch_size
        self.out_number = cfgs.model.layer_sample 
        self.temperature = cfgs.loss_temperature
        self.loss_layerchoice = cfgs.model.loss_layerchoice
        self.device = args.device
        self.sample_strategy = cfgs.model.sample_strategy
        self.depth = cfgs.model.depth
        self.distance_measure = cfgs.model.distance_measure
        self.criterion1 = nn.CrossEntropyLoss(reduction="mean")
        self.criterion2 = nn.L1Loss(reduction='mean')
        self.fea_weight_choice = cfgs.model.fea_weight_choice
        self.bcl_multi_layer_choice = cfgs.model.bcl_multi_layer_choice

        self.similarityfirst = cfgs.model.similarityfirst_encoder
        self.patchnumber = cfgs.model.patchnumber
        self.patchforeground_weight = cfgs.model.patchforeground_weight

        self.efs = 1e-20


    def get_weight(self, weight):
        mask = torch.ones(self.batch_size, weight.size(1), 1).to(self.device)
        weight1 = torch.matmul(mask, weight.unsqueeze(1))
        weight2 = weight1 - weight.unsqueeze(2)
        denominator = abs(weight2)
        numerator = 1 - denominator
        return numerator, denominator

    def cal_loss(self, a1, gt, number):  
        loss_eachlayer = 0

        if self.similarityfirst == True:
            for i in range(gt.size()[1]): 

                gt_ =  gt[:,i,:]
                numerator, denominator = self.get_weight(gt_)
                if self.distance_measure == 'cosine':
                    similarity = nn.CosineSimilarity(dim=1)(a1.unsqueeze(3), a1.unsqueeze(2)) / self.temperature  

                number = similarity.size()[1]

                for criterion_name, criterion_weight in self.cfgs.criterions:
                    if criterion_name == 'encoder_patchseries':
                        similarity = similarity - torch.max(similarity)
                        positive0 = torch.mul(similarity, denominator)
                        positive1 = torch.exp(positive0)
                        negative0 = torch.mul(similarity, numerator)
                        negative1 = torch.exp(negative0)
                        negative2 = torch.sum(negative1, 2).unsqueeze(2)   
                        negative3 = torch.repeat_interleave(negative2, repeats=negative0.size()[1], dim=2)
                        logits = - torch.log( positive1 / negative3).sum()
                        loss = criterion_weight * logits
                        loss_eachlayer += loss

            return loss_eachlayer / self.patchnumber
        
        elif self.similarityfirst == False:
            patch_weight = torch.repeat_interleave(torch.tensor(self.patchforeground_weight).unsqueeze(0), self.batch_size, dim=0).to(self.device)
            gt_ = torch.mul(gt, patch_weight.unsqueeze(2))
            gt_ = torch.sum(gt_, dim = 1) / sum(self.patchforeground_weight)
            numerator, denominator = self.get_weight(gt_)

            if self.distance_measure == 'cosine':
                #print(type(a1))
                similarity = nn.CosineSimilarity(dim=1)(a1.unsqueeze(3), a1.unsqueeze(2)) / self.temperature  
            number = similarity.size()[1]
            for criterion_name, criterion_weight in self.cfgs.criterions:
                if criterion_name == 'encoder_patchseries':
                    similarity = similarity - torch.max(similarity)
                    positive0 = torch.mul(similarity, denominator)
                    positive1 = torch.exp(positive0)
                    negative0 = torch.mul(similarity, numerator)
                    negative1 = torch.exp(negative0)
                    negative2 = torch.sum(negative1, 2).unsqueeze(2) 
                    negative3 = torch.repeat_interleave(negative2, repeats=negative0.size()[1], dim=2)
                    logits = - torch.log( positive1 / negative3).sum()

                    loss = criterion_weight * logits
                    loss_eachlayer += loss 

            return loss_eachlayer


    def forward(self, pred, a):
        loss = 0
        f = pred[1]
        gt = pred[2]
        for i in range(len(f)):
            tem_loss = self.cal_loss(f[i], gt[i], self.out_number[i])
            loss += tem_loss

        return loss 
