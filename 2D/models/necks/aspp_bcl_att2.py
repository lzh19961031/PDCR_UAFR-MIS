import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .get_uncertainty_att import *

EFS = 1e-10

__all__ = ['build_aspp_bcl_att2']


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()


    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                #if m.bias:
                m.bias.data.zero_()


class _ASPPModulegt(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModulegt, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.fill_(1)
                if m.bias:
                    m.bias.data.zero_()


class ASPP_bcl_att2(nn.Module):
    def __init__(self, cfgs, backbone, output_stride, BatchNorm):
        super(ASPP_bcl_att2, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.fea_weight_choice = 'max'
        
        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.imgaspplist = nn.ModuleList([self.global_avg_pool, self.conv1, self.bn1])
        self._init_imgaspp_weight()

        self.uncertainty_att_layerchoice = cfgs.model.uncertainty_att_layerchoice
        if self.uncertainty_att_layerchoice[5] == 1:
            self.conv_uncertainty1 = nn.Sequential(nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(16),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),
                                        nn.Conv2d(16, 2, kernel_size=1, stride=1))
        if self.uncertainty_att_layerchoice[6] == 1:                                
            self.conv_uncertainty2 = nn.Sequential(nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(16),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),
                                        nn.Conv2d(16, 2, kernel_size=1, stride=1))
        if self.uncertainty_att_layerchoice[7] == 1:                                
            self.conv_uncertainty3 = nn.Sequential(nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(16),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),
                                        nn.Conv2d(16, 2, kernel_size=1, stride=1))
        if self.uncertainty_att_layerchoice[8] == 1:                                
            self.conv_uncertainty4 = nn.Sequential(nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(16),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),
                                        nn.Conv2d(16, 2, kernel_size=1, stride=1))
        if self.uncertainty_att_layerchoice[9] == 1:                                
            self.conv_uncertainty = nn.Sequential(nn.Conv2d(1280, 16, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(16),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),
                                        nn.Conv2d(16, 2, kernel_size=1, stride=1))

        self.uncertainty_output = []
        self.uncertainty_gt = []
        self.softmax = nn.Softmax(dim=1)

        self.aspp1gt = _ASPPModulegt(1, 1,  1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2gt = _ASPPModulegt(1, 1,  3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3gt = _ASPPModulegt(1, 1,  3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4gt = _ASPPModulegt(1, 1,  3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_poolgt = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(1, 1, 1, stride=1, bias=False))
        self.conv1gt = nn.Conv2d(5, 1, 1, bias=False)

        self.maskaspplist = nn.ModuleList([self.global_avg_poolgt, self.conv1gt])
        self._init_maskaspp_weight()


    def forward(self, x, mask):
        x1 = self.aspp1(x)
        if self.uncertainty_att_layerchoice[5] == 1:
            x1, uncertainty_output1, uncertainty_gt1 = get_uncertainty_att(x1, mask, self.conv_uncertainty1)
            self.uncertainty_output.append(uncertainty_output1)
            self.uncertainty_gt.append(uncertainty_gt1)

        x2 = self.aspp2(x)
        if self.uncertainty_att_layerchoice[6] == 1:
            x2, uncertainty_output2, uncertainty_gt2 = get_uncertainty_att(x2, mask, self.conv_uncertainty2)
            self.uncertainty_output.append(uncertainty_output2)
            self.uncertainty_gt.append(uncertainty_gt2)

        x3 = self.aspp3(x)
        if self.uncertainty_att_layerchoice[7] == 1:
            x3, uncertainty_output3, uncertainty_gt3 = get_uncertainty_att(x3, mask, self.conv_uncertainty3)
            self.uncertainty_output.append(uncertainty_output3)
            self.uncertainty_gt.append(uncertainty_gt3)

        x4 = self.aspp4(x)
        if self.uncertainty_att_layerchoice[8] == 1:
            x4, uncertainty_output4, uncertainty_gt4 = get_uncertainty_att(x4, mask, self.conv_uncertainty4)
            self.uncertainty_output.append(uncertainty_output4)
            self.uncertainty_gt.append(uncertainty_gt4)

        x5_ = self.global_avg_pool(x)
        x5 = F.interpolate(x5_, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        if self.uncertainty_att_layerchoice[9] == 1:
            x, uncertainty_output, uncertainty_gt = get_uncertainty_att(x, mask, self.conv_uncertainty)
            self.uncertainty_output.append(uncertainty_output)
            self.uncertainty_gt.append(uncertainty_gt)

            
        x = self.conv1(x)
        x = self.bn1(x)
        x_ = self.relu(x)
        x = self.dropout(x_)

        if self.fea_weight_choice == 'max':
            x1_mask = self.aspp1gt(mask)
            x1_mask = x1_mask / (torch.max(x1_mask)+EFS)

            x2_mask = self.aspp2gt(mask)
            x2_mask = x2_mask / (torch.max(x2_mask)+EFS)

            x3_mask = self.aspp3gt(mask)
            x3_mask = x3_mask / (torch.max(x3_mask)+EFS)

            x4_mask = self.aspp4gt(mask)
            x4_mask = x4_mask / (torch.max(x4_mask)+EFS)

            x5__mask = self.global_avg_poolgt(mask)
            x5__mask = x5__mask / (torch.max(x5__mask)+EFS)

            x5_mask = F.interpolate(x5__mask, size=x4_mask.size()[2:], mode='bilinear', align_corners=True)
            mask = torch.cat((x1_mask, x2_mask, x3_mask, x4_mask, x5_mask), dim=1)

            x__mask = self.conv1gt(mask)
            mask = x__mask / (torch.max(x__mask)+EFS)

        return [x1, x2, x3, x4, x5_, x_], [x1_mask, x2_mask, x3_mask, x4_mask, x5__mask, x__mask], x, mask, self.uncertainty_output, self.uncertainty_gt

    def _init_imgaspp_weight(self):
        for m in self.imgaspplist:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                #if m.bias:
                m.bias.data.zero_()


    def _init_maskaspp_weight(self):
        for m in self.maskaspplist:
            if isinstance(m, nn.Conv2d):
                m.weight.data.fill_(1)
                if m.bias:
                    m.bias.data.zero_()


def build_aspp_bcl_att2(cfgs, backbone, output_stride, BatchNorm):
    return ASPP_bcl_att2(cfgs, backbone, output_stride, BatchNorm)  
