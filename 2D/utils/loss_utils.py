import torch
import torch.nn as nn
import sys
import numpy as np
sys.path.append('../')
from criterions import dice, bcl, scl, multi_layer_cl, decoder_bcl_rf


class get_criterion(nn.Module):

    def __init__(self, args, cfgs):
        self.args = args
        self.cfgs = cfgs
        self.criterion_collection = []
        class_weight = torch.tensor(np.array(cfgs.class_weight)).float().to(args.device)

        for criterion_name, criterion_weight in cfgs.criterions:              
            if criterion_name == 'ce':
                criterion = nn.CrossEntropyLoss(weight=class_weight, reduction='mean')
            elif criterion_name == 'encoder_patchseries':
                criterion = bcl.CL_Loss(self.args, self.cfgs)
            elif criterion_name == 'decoder_patchseries':
                criterion = decoder_bcl_rf.CL_Loss(self.args, self.cfgs)
            elif 'multi_layer' in criterion_name:
                criterion = multi_layer_cl.Multi_layer_CL_Loss(self.args, self.cfgs)
            else:
                print('loss not exist!')
                raise NotImplementedError
            self.criterion_collection.append((criterion, criterion_weight, criterion_name))

    def forward(self, pred, gt):
        final_loss = 0
        if type(pred) == tuple or type(pred) == list:
            output = pred[0]
        else:
            output = pred

        for item in self.criterion_collection:       
            criterion, criterion_weight, loss_name = item
            if loss_name == 'ce':
                gt = gt.squeeze(1)
                cur_loss = criterion_weight * criterion(output, gt)
                final_loss += cur_loss

            elif 'multi_layer' in loss_name:
                cur_loss = criterion(pred, gt)
                final_loss += cur_loss     

            elif loss_name == 'encoder_patchseries':
                cur_loss = criterion(pred, gt)
                final_loss += cur_loss

            elif loss_name == 'decoder_patchseries':
                cur_loss = criterion(pred, gt)
                final_loss += cur_loss

            else:
                print('loss not exist!')
                raise NotImplementedError

        return final_loss

