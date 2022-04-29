import torch
from random import sample
import random
import torch.nn as nn
import yaml
from addict import Dict
from itertools import combinations, permutations

class Multi_layer_CL_Loss(torch.nn.Module):

    def __init__(self, args, cfgs):
        super(Multi_layer_CL_Loss, self).__init__()
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

        self.similarityfirst = cfgs.model.similarityfirst_multi
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


    def cal_multilayer_loss(self, a, gt, bcl_multi_layer_choice):  
        loss_multilayer = 0
        if self.similarityfirst == True:
            for i in range(len(a)): 
                for j in range(i+1, len(a)):
                    for k in range(gt[i].size()[1]): 
                        if self.distance_measure == 'cosine':
                            a1 = a[i]
                            a2 = a[j]
                            similarity = nn.CosineSimilarity(dim=1)(a1.unsqueeze(3), a2.unsqueeze(2)) / self.temperature  
                            gt1 = gt[i]
                            gt2 = gt[j]
                            gt1_ =  gt1[:,k,:]
                            gt2_ =  gt2[:,k,:]
                            numerator, denominator = self.get_weight_multi_layer(gt1_, gt2_) 
                            for criterion_name, criterion_weight in self.cfgs.criterions:
                                if 'multi_layer' in criterion_name:
                                    similarity = similarity - torch.max(similarity)
                                    positive0 = torch.mul(similarity, denominator)
                                    positive1 = torch.exp(positive0)
                                    negative0 = torch.mul(similarity, numerator)
                                    negative1 = torch.exp(negative0)
                                    negative2 = torch.sum(negative1, 2).unsqueeze(2)   
                                    negative3 = torch.repeat_interleave(negative2, repeats=negative0.size()[1], dim=2)
                                    logits = - torch.log( positive1 / negative3).sum()
                                    loss = criterion_weight * logits 
                                    loss_multilayer += loss
            return loss_multilayer / self.patchnumber 


        elif self.similarityfirst == False:
            for i in range(len(a)): 
                for j in range(i+1, len(a)):
                    if self.distance_measure == 'cosine':
                        a1 = a[i]
                        a2 = a[j]
                        similarity = nn.CosineSimilarity(dim=1)(a1.unsqueeze(3), a2.unsqueeze(2)) / self.temperature  
                        gt1 = gt[i]
                        gt2 = gt[j]
                        patch_weight = torch.repeat_interleave(torch.tensor(self.patchforeground_weight).unsqueeze(0), self.batch_size, dim=0).to(self.device)
                        gt1_ = torch.mul(gt1, patch_weight.unsqueeze(2))
                        gt1_ = torch.sum(gt1_, dim = 1) / sum(self.patchforeground_weight)
                        gt2_ = torch.mul(gt2, patch_weight.unsqueeze(2))
                        gt2_ = torch.sum(gt2_, dim = 1) / sum(self.patchforeground_weight)
                        numerator, denominator = self.get_weight_multi_layer(gt1_, gt2_) 
                        for criterion_name, criterion_weight in self.cfgs.criterions:
                            if 'multi_layer' in criterion_name:
                                similarity = similarity - torch.max(similarity)
                                positive0 = torch.mul(similarity, denominator)
                                positive1 = torch.exp(positive0)
                                negative0 = torch.mul(similarity, numerator)
                                negative1 = torch.exp(negative0)
                                negative2 = torch.sum(negative1, 2).unsqueeze(2)  
                                negative3 = torch.repeat_interleave(negative2, repeats=negative0.size()[1], dim=2)
                                logits = - torch.log( positive1 / negative3).sum()
                                loss = criterion_weight * logits 
                                loss_multilayer += loss
            return loss_multilayer / self.patchnumber 

    def forward(self, pred, a):
        loss = 0
        f_multilayer = pred[5]
        gt_multilayer = pred[6]
        multi_layer_loss = self.cal_multilayer_loss(f_multilayer, gt_multilayer, self.bcl_multi_layer_choice)
        loss += multi_layer_loss

        return loss 

