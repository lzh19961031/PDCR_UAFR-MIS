import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from addict import Dict
from .necks.aspp_bcl_att2 import build_aspp_bcl_att2
from .decoders.decoder_bcl_att2 import build_decoder_bcl_att2
from .backbones import build_backbone
import yaml
from addict import Dict
from .receptivefield import *
from .sample_by_rf import *
from .calculate_patch_max_length import *


__all__ = ['DeepLab_bcl_att2']


class DeepLab_bcl_att2(nn.Module):
    def __init__(self, args, cfgs):
        super(DeepLab_bcl_att2, self).__init__()

        backbone = cfgs.model.backbone
        output_stride = cfgs.model.output_stride
        freeze_bn = cfgs.model.freeze_bn
        num_classes = cfgs.model.num_classes

        if backbone == 'drn':
            output_stride = 8

        BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(cfgs, backbone, output_stride, BatchNorm)
        self.aspp = build_aspp_bcl_att2(cfgs, backbone, output_stride, BatchNorm)
        self.decoder = build_decoder_bcl_att2(cfgs, num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn
        self.loss_layerchoice = cfgs.model.loss_layerchoice
        self.fea_weight_choice = cfgs.model.fea_weight_choice

        self.input_channel = cfgs.model.input_channel
        self.inputsize0 = cfgs.dataset.input_size[0]
        self.inputsize1 = cfgs.dataset.input_size[1]
        self.device = args.device

        self.backbonetype = cfgs.model.backbone
        self.backbone_rf = build_backbone(cfgs, 'xception_bcl_decoder_rf', output_stride, BatchNorm)
        self.receptive_field_dict, self.qualified_fea_number = receptivefield(self.backbone_rf.to(self.device), (1, self.input_channel, self.inputsize0, self.inputsize1), (1, 1, self.inputsize0, self.inputsize1), device=self.device)
        self.out_number = cfgs.model.layer_sample
        self.out_number_multilayer = cfgs.model.layer_sample_multilayer
        self.batch_size = cfgs.batch_size
        self.instance_dis = cfgs.model.instance_dis
        self.bcl_multi_layer_choice = cfgs.model.bcl_multi_layer_choice

        self.patchnumber = int(cfgs.model.patchnumber)
        self.patchsize = cfgs.model.patchsize
        self.patch_max_length = max(self.patchsize)
        self.similarityfirst = cfgs.model.similarityfirst


        self.instance_dis = cfgs.model.instance_dis
        if cfgs.model.bcl_layer_choice:
            self.bcl_layer_switch = True 
            self.bcl_layer = self.qualified_fea_number * [0]
            for i in range(len(cfgs.model.bcl_layer_choice)):
                self.bcl_layer[cfgs.model.bcl_layer_choice[i]-1] = 1
        else:
            self.bcl_layer_switch = False
        self.bcl_multi_layer = self.qualified_fea_number * [0]
        for i in range(len(cfgs.model.bcl_multi_layer_choice)):
            self.bcl_multi_layer[cfgs.model.bcl_multi_layer_choice[i]-1] = 1
        self.center_location_in_orimask = []
        for i in range(self.qualified_fea_number):
            center_location = cal_center_location_in_orimask(self.receptive_field_dict[i], self.patch_max_length, self.inputsize0, self.inputsize1, self.device)
            self.center_location_in_orimask.append(center_location)

    def forward(self, input, mask):
        mask = mask.float()
        hid_fea1, hid_fea_multilayer, x, low_level_feat, x_mask, low_level_feat_mask, uncertainty_output1, uncertainty_gt1 = self.backbone(input, mask)
        hid_fea2, hid_gtweight2, x, x_mask, uncertainty_output2, uncertainty_gt2 = self.aspp(x, mask)
        hid_fea3, hid_gtweight3, x, uncertainty_output3, uncertainty_gt3 = self.decoder(x, low_level_feat, mask)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        hid_fea = hid_fea1
        hid_fea_decoder = hid_fea3

        if self.fea_weight_choice == 'max':
            hid_gtweight = hid_gtweight1 + hid_gtweight2 + hid_gtweight3     
            hid_fea, hid_gtweight = samplefor_strategymax(hid_fea, hid_gtweight, a.size()[2], number)
            return x, hid_fea, hid_gtweight
        
        elif self.fea_weight_choice == 'receptive_field':
            hid_feature_ = []
            hid_feature_decoder_ = []
            hid_gtweight = []
            hid_gtweight_decoder = []

            if self.bcl_layer_switch == True and self.instance_dis == False:
                for i in range(len(self.bcl_layer)):
                    if self.bcl_layer[i] == 1:
                        hidfea = hid_fea[i]
                        y_min = int(self.patch_max_length)
                        y_max = int(hidfea.size()[2]-self.patch_max_length)
                        x_min = int(self.patch_max_length)
                        x_max = int(hidfea.size()[3]-self.patch_max_length)
                        hidfea_ = hidfea[:, :, y_min:y_max, x_min:x_max]

                        sampleindex, sampleindex_x, sampleindex_y = sample_index(hidfea_.size()[2], hidfea_.size()[3], self.out_number[i])
                        hid_feature = samplefor_strategyrf(hidfea_, sampleindex_y, sampleindex_x)
                        hid_feature_.append(hid_feature)
                        hid_gt = cal_numberof_one(mask, self.patchsize, self.patchnumber , self.inputsize0, self.inputsize1, sampleindex, self.center_location_in_orimask[i], hidfea_.size()[2], hidfea_.size()[3], self.batch_size, self.device)
                        hid_gtweight.append(hid_gt)
                hidfea_decoder = hid_fea_decoder[0]
                y_min = int(self.patch_max_length)
                y_max = int(hidfea_decoder.size()[2]-self.patch_max_length)
                x_min = int(self.patch_max_length)
                x_max = int(hidfea_decoder.size()[3]-self.patch_max_length)
                hidfea_decoder_ = hidfea_decoder[:, :, y_min:y_max, x_min:x_max]                
                sampleindex, sampleindex_x, sampleindex_y = sample_index(hidfea_decoder_.size()[2], hidfea_decoder_.size()[3], self.out_number[2])
                hid_feature_decoder = samplefor_strategyrf(hidfea_decoder_, sampleindex_y, sampleindex_x)
                hid_feature_decoder_.append(hid_feature_decoder)
                hid_gt_decoder = cal_numberof_one(mask, self.patchsize, self.patchnumber, self.inputsize0, self.inputsize1, sampleindex, self.center_location_in_orimask[-2], hidfea_decoder_.size()[2], hidfea_decoder_.size()[3], self.batch_size, self.device)
                hid_gtweight_decoder.append(hid_gt_decoder)
                hidfea_decoder = hid_fea_decoder[1]
                y_min = int(self.patch_max_length)
                y_max = int(hidfea_decoder.size()[2]-self.patch_max_length)
                x_min = int(self.patch_max_length)
                x_max = int(hidfea_decoder.size()[3]-self.patch_max_length)
                hidfea_decoder_ = hidfea_decoder[:, :, y_min:y_max, x_min:x_max]                
                sampleindex, sampleindex_x, sampleindex_y = sample_index(hidfea_decoder_.size()[2], hidfea_decoder_.size()[3], self.out_number[3])
                hid_feature_decoder = samplefor_strategyrf(hidfea_decoder_, sampleindex_y, sampleindex_x)
                hid_feature_decoder_.append(hid_feature_decoder)
                hid_gt_decoder = cal_numberof_one(mask, self.patchsize, self.patchnumber , self.inputsize0, self.inputsize1, sampleindex, self.center_location_in_orimask[-1], hidfea_decoder_.size()[2], hidfea_decoder_.size()[3], self.batch_size, self.device)
                hid_gtweight_decoder.append(hid_gt_decoder)
            else:
                raise ValueError('既不是所有qualified feature都bcl，也不是选层bcl，也不是bcl+后面层instance_dis')
            hid_feature_multilayer_ = []
            hid_gtweight_multilayer = []
            for i in range(len(hid_fea_multilayer)):
                if self.bcl_multi_layer[i] == 1:
                    hidfea_multilayer = hid_fea_multilayer[i]
                    y_min = int(self.patch_max_length)
                    y_max = int(hidfea_multilayer.size()[2]-self.patch_max_length)
                    x_min = int(self.patch_max_length)
                    x_max = int(hidfea_multilayer.size()[3]-self.patch_max_length)
                    sampleindex, sampleindex_x, sampleindex_y = sample_index(hidfea_multilayer.size()[2], hidfea_multilayer.size()[3], self.out_number_multilayer[i])
                    hid_feature_multilayer = samplefor_strategyrf(hidfea_multilayer, sampleindex_y, sampleindex_x)
                    hid_feature_multilayer_.append(hid_feature_multilayer)
                    hid_gt_multilayer = cal_numberof_one(mask, self.patchsize, self.patchnumber, self.inputsize0, self.inputsize1, sampleindex, self.center_location_in_orimask[i], hidfea_multilayer.size()[2], hidfea_multilayer.size()[3], self.batch_size, self.device)
                    hid_gtweight_multilayer.append(hid_gt_multilayer)
            return x, hid_feature_, hid_gtweight, hid_feature_decoder_, hid_gtweight_decoder, hid_feature_multilayer_, hid_gtweight_multilayer

    def apply_freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


