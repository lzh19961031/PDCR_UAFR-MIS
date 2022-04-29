import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .get_uncertainty_att import *

__all__ = ['build_decoder_bcl_att2']


EFS = 1e-10

class Decoder_bcl_att2(nn.Module):
    def __init__(self, cfgs, num_classes, backbone, BatchNorm):
        super(Decoder_bcl_att2, self).__init__()
        if 'resnet' in backbone or 'drn' in backbone:
            low_level_inplanes = 256
        elif 'xception' in backbone:
            low_level_inplanes = 128
        elif 'mobilenet' in backbone:
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.fea_weight_choice = 'max'

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv1 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5))
        self.last_conv2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1))
        self.last_conv3 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)


        self.imgdecoderlist = nn.ModuleList([self.conv1, self.bn1, self.last_conv1, self.last_conv2])
        self._init_imgdecoder_weight()      
        
        self.uncertainty_att_layerchoice = cfgs.model.uncertainty_att_layerchoice
        if self.uncertainty_att_layerchoice[10] == 1:
            self.conv_uncertainty_low_level_feat = nn.Sequential(nn.Conv2d(48, 16, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(16),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),
                                        nn.Conv2d(16, 2, kernel_size=1, stride=1))
        if self.uncertainty_att_layerchoice[11] == 1:                                
            self.conv_uncertainty = nn.Sequential(nn.Conv2d(304, 16, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(16),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),
                                        nn.Conv2d(16, 2, kernel_size=1, stride=1))
        self.uncertainty_output = []
        self.uncertainty_gt = []
        self.softmax = nn.Softmax(dim=1)

        self.conv1gt = nn.Conv2d(1, 1, 1, bias=False)      
        self.last_conv1gt = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.last_conv2gt = nn.Sequential(nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.Conv2d(1, 1, kernel_size=1, stride=1))

        self.maskdecoderlist = nn.ModuleList([self.conv1gt, self.last_conv1gt, self.last_conv2gt])
        self._init_maskdecoder_weight()



    def forward(self, x, low_level_feat, mask):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        if self.uncertainty_att_layerchoice[10] == 1:
            low_level_feat, uncertainty_output_low_level_feat, uncertainty_gt_low_level_feat = get_uncertainty_att(low_level_feat, mask, self.conv_uncertainty_low_level_feat)
            self.uncertainty_output.append(uncertainty_output_low_level_feat)
            self.uncertainty_gt.append(uncertainty_gt_low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        if self.uncertainty_att_layerchoice[11] == 1:
            x, uncertainty_output, uncertainty_gt = get_uncertainty_att(x, mask, self.conv_uncertainty)            
            self.uncertainty_output.append(uncertainty_output)
            self.uncertainty_gt.append(uncertainty_gt)

        x1 = self.last_conv1(x)
        x2 = self.last_conv2(x1)
        x = self.last_conv3(x2)

        if self.fea_weight_choice == 'max':        
            mask = self.conv1gt(mask)
            mask = mask / (torch.max(mask)+EFS)

            x_mask = F.interpolate(mask, size=mask.size()[2:], mode='bilinear', align_corners=True)
            x_mask = x_mask / (torch.max(x_mask)+EFS)

            x_mask = torch.cat((x_mask, mask), dim=1)
            x1_mask = self.last_conv1gt(x_mask)
            x1_mask = x1_mask / (torch.max(x1_mask)+EFS)

            x_mask = self.last_conv2gt(x1_mask)

        return [low_level_feat, x1], [x1, x2], x, self.uncertainty_output, self.uncertainty_gt

    def _init_imgdecoder_weight(self):
        for m in self.imgdecoderlist:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                #if m.bias:
                m.bias.data.zero_()

    def _init_maskdecoder_weight(self):
        for m in self.maskdecoderlist:
            if isinstance(m, nn.Conv2d):
                m.weight.data.fill_(1)
                if m.bias:
                    m.bias.data.zero_()


def build_decoder_bcl_att2(cfgs, num_classes, backbone, BatchNorm):
    return Decoder_bcl_att2(cfgs, num_classes, backbone, BatchNorm)