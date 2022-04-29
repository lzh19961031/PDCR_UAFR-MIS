import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from .get_uncertainty_att import *

EFS = 1e-10


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, BatchNorm=None):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                               groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class SeparableConv2dgt(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, BatchNorm=None):
        super(SeparableConv2dgt, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                               groups=inplanes, bias=bias)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

        self._init_weight()

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.fill_(1)
                if m.bias:
                    m.bias.data.zero_()



class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, BatchNorm=None,
                 start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            self.skipbn = BatchNorm(planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 2, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        if stride == 1 and is_last:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 1, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        if not start_with_relu:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

        self._init_weight()

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = x + skip

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Blockgt(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, BatchNorm=None,
                 start_with_relu=False, grow_first=True, is_last=False):
        super(Blockgt, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
        else:
            self.skip = None
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(SeparableConv2dgt(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm))
            filters = planes

        for i in range(reps - 1):
            rep.append(SeparableConv2dgt(filters, filters, 3, 1, dilation, BatchNorm=BatchNorm))

        if not grow_first:
            rep.append(SeparableConv2dgt(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm))

        if stride != 1:
            rep.append(SeparableConv2dgt(planes, planes, 3, 2, BatchNorm=BatchNorm))

        if stride == 1 and is_last:
            rep.append(SeparableConv2dgt(planes, planes, 3, 1, BatchNorm=BatchNorm))

        if not start_with_relu:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

        self._init_weight()

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
        else:
            skip = inp

        x = x + skip

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.fill_(1)
                if m.bias:
                    m.bias.data.zero_()


class AlignedXception_bcl_att2(nn.Module):
    """
    Modified Alighed Xception
    """

    def __init__(self, cfgs, output_stride, BatchNorm, pretrained=True):
        super(AlignedXception_bcl_att2, self).__init__()

        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError

        self.fea_weight_choice = 'max'


        self.conv1_multilayerloss = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=8, stride=8, padding=0, bias=False),
            BatchNorm(32),
            nn.ReLU())           
        self.conv2_multilayerloss = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=8, stride=8, padding=0, bias=False),
            BatchNorm(32),
            nn.ReLU())                
        self.conv3_multilayerloss = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=4, stride=4, padding=0, bias=False),
            BatchNorm(32),
            nn.ReLU())    
        self.conv4_multilayerloss = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=2, stride=2, padding=0, bias=False),
            BatchNorm(32),
            nn.ReLU())  
        self.conv5_multilayerloss = nn.Sequential(
            nn.Conv2d(728, 32, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(32),
            nn.ReLU())  

        self.uncertainty_att_layerchoice = cfgs.model.uncertainty_att_layerchoice
        if self.uncertainty_att_layerchoice[0] == 1:
            self.conv_uncertainty1 = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(16),
                                        nn.ReLU(),
                                        #    nn.Dropout(0.1),
                                        nn.Conv2d(16, 2, kernel_size=1, stride=1))
        if self.uncertainty_att_layerchoice[1] == 1:
            self.conv_uncertainty2 = nn.Sequential(nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(16),
                                        nn.ReLU(),
                                        #    nn.Dropout(0.1),
                                        nn.Conv2d(16, 2, kernel_size=1, stride=1))
        if self.uncertainty_att_layerchoice[2] == 1:
            self.conv_uncertainty5 = nn.Sequential(nn.Conv2d(1536, 16, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(16),
                                        nn.ReLU(),
                                        #    nn.Dropout(0.1),
                                        nn.Conv2d(16, 2, kernel_size=1, stride=1))
        if self.uncertainty_att_layerchoice[3] == 1:                                
            self.conv_uncertainty6 = nn.Sequential(nn.Conv2d(1536, 16, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(16),
                                        nn.ReLU(),
                                        #    nn.Dropout(0.1),
                                        nn.Conv2d(16, 2, kernel_size=1, stride=1))
        if self.uncertainty_att_layerchoice[4] == 1:
            self.conv_uncertainty = nn.Sequential(nn.Conv2d(2048, 16, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(16),
                                        nn.ReLU(),
                                        #    nn.Dropout(0.1),
                                        nn.Conv2d(16, 2, kernel_size=1, stride=1))
        self.uncertainty_output = []
        self.uncertainty_gt = []
        self.softmax = nn.Softmax(dim=1)

        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm(64)

        self.block1 = Block(64, 128, reps=2, stride=2, BatchNorm=BatchNorm, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, BatchNorm=BatchNorm, start_with_relu=False,
                            grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride, BatchNorm=BatchNorm,
                            start_with_relu=True, grow_first=True, is_last=True)

        self.block4  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block5  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block6  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block7  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block8  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block9  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block13 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block14 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block15 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block16 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block17 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block18 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block19 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)

        self.block20 = Block(728, 1024, reps=2, stride=1, dilation=exit_block_dilations[0],
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=False, is_last=True)

        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn3 = BatchNorm(1536)

        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn4 = BatchNorm(1536)

        self.conv5 = SeparableConv2d(1536, 2048, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn5 = BatchNorm(2048)

        self.imgxceptionlist = nn.ModuleList([self.conv1, self.bn1, self.conv2, self.bn2, self.conv3, self.bn3, self.conv4, self.bn4, self.conv5, self.bn5])
        self._init_imgxception_weight()



        ###############################################################################################################
        # gt weight 
        ###############################################################################################################
        # Entry flow
        self.conv1gt = nn.Conv2d(1, 1, 3, stride=2, padding=1, bias=False)
        #self.bn1gt = BatchNorm(1)
        #self.relugt = nn.ReLU(inplace=True)

        self.conv2gt = nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False)
        #self.bn2gt = BatchNorm(1)

        self.block1gt = Blockgt(1, 1, reps=2, stride=2, BatchNorm=BatchNorm, start_with_relu=False)
        self.block2gt = Blockgt(1, 1, reps=2, stride=2, BatchNorm=BatchNorm, start_with_relu=False,
                            grow_first=True)
        self.block3gt = Blockgt(1, 1, reps=2, stride=entry_block3_stride, BatchNorm=BatchNorm,
                            start_with_relu=False, grow_first=True, is_last=True)

        # Middle flow
        self.block4gt  = Blockgt(1, 1, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=False, grow_first=True)
        self.block5gt  = Blockgt(1, 1, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=False, grow_first=True)
        self.block6gt  = Blockgt(1, 1, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=False, grow_first=True)
        self.block7gt  = Blockgt(1, 1, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=False, grow_first=True)
        self.block8gt  = Blockgt(1, 1, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=False, grow_first=True)
        self.block9gt  = Blockgt(1, 1, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=False, grow_first=True)
        self.block10gt = Blockgt(1, 1, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=False, grow_first=True)
        self.block11gt = Blockgt(1, 1, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=False, grow_first=True)
        self.block12gt = Blockgt(1, 1, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=False, grow_first=True)
        self.block13gt = Blockgt(1, 1, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=False, grow_first=True)
        self.block14gt = Blockgt(1, 1, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=False, grow_first=True)
        self.block15gt = Blockgt(1, 1, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=False, grow_first=True)
        self.block16gt = Blockgt(1, 1, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=False, grow_first=True)
        self.block17gt = Blockgt(1, 1, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=False, grow_first=True)
        self.block18gt = Blockgt(1, 1, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=False, grow_first=True)
        self.block19gt = Blockgt(1, 1, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=False, grow_first=True)

        # Exit flow
        self.block20gt = Blockgt(1, 1, reps=2, stride=1, dilation=exit_block_dilations[0],
                             BatchNorm=BatchNorm, start_with_relu=False, grow_first=False, is_last=True)

        self.conv3gt = SeparableConv2dgt(1, 1, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        #self.bn3gt = BatchNorm(1)

        self.conv4gt = SeparableConv2dgt(1, 1, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        #self.bn4gt = BatchNorm(1)

        self.conv5gt = SeparableConv2dgt(1, 1, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        #self.bn5gt = BatchNorm(1)
   
        self.maskxceptionlist = nn.ModuleList([self.conv1gt, self.conv2gt, self.conv3gt, self.conv4gt, self.conv5gt])
        self._init_maskxception_weight()

        # Load pretrained model
        if pretrained:
            self._load_pretrained_model()

    def forward(self, x, mask):
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x1_multilayer_loss = self.conv1_multilayerloss(x1)
        if self.uncertainty_att_layerchoice[0] == 1:
            x1, uncertainty_output1, uncertainty_gt1 = get_uncertainty_att(x1, mask, self.conv_uncertainty1)
            self.uncertainty_output.append(uncertainty_output1)
            self.uncertainty_gt.append(uncertainty_gt1)

        x = self.conv2(x1)
        x = self.bn2(x)
        x2 = self.relu(x)
        x2_multilayer_loss = self.conv2_multilayerloss(x2)
        if self.uncertainty_att_layerchoice[1] == 1:
            x2, uncertainty_output2, uncertainty_gt2 = get_uncertainty_att(x2, mask, self.conv_uncertainty2)
            self.uncertainty_output.append(uncertainty_output2)
            self.uncertainty_gt.append(uncertainty_gt2)

        x3 = self.block1(x2)
        x3 = self.relu(x3)
        x3_multilayer_loss = self.conv3_multilayerloss(x3)
        low_level_feat = x3
        x4 = self.block2(x3)
        x4_multilayer_loss = self.conv4_multilayerloss(x4)
        x5 = self.block3(x4)
        x5_multilayer_loss = self.conv5_multilayerloss(x5)

        x6 = self.block4(x5)
        x7 = self.block5(x6)
        x8 = self.block6(x7)
        x9 = self.block7(x8)
        x = self.block8(x9)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x10 = self.block19(x)

        x = self.block20(x10)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x11 = self.relu(x)
        if self.uncertainty_att_layerchoice[2] == 1:
            x11, uncertainty_output5, uncertainty_gt5 = get_uncertainty_att(x11, mask, self.conv_uncertainty5)
            self.uncertainty_output.append(uncertainty_output5)
            self.uncertainty_gt.append(uncertainty_gt5)

        x = self.conv4(x11)
        x = self.bn4(x)
        x6 = self.relu(x)
        if self.uncertainty_att_layerchoice[3] == 1:
            x6, uncertainty_output6, uncertainty_gt6 = get_uncertainty_att(x6, mask, self.conv_uncertainty6)
            self.uncertainty_output.append(uncertainty_output6)
            self.uncertainty_gt.append(uncertainty_gt6)

        x = self.conv5(x6)
        x = self.bn5(x)
        x = self.relu(x)
        if self.uncertainty_att_layerchoice[4] == 1:
            x, uncertainty_output, uncertainty_gt = get_uncertainty_att(x, mask, self.conv_uncertainty)
            self.uncertainty_output.append(uncertainty_output)
            self.uncertainty_gt.append(uncertainty_gt)

        if self.fea_weight_choice == 'max':
            x1_mask = self.conv1gt(mask)
            x1_mask = x1_mask / (torch.max(x1_mask)+EFS)

            x2_mask = self.conv2gt(x1_mask)
            x2_mask = x2_mask / (torch.max(x2_mask)+EFS)

            mask = self.block1gt(x2_mask)
            low_level_feat_mask = mask
            mask = self.block2gt(mask)
            x3_mask = self.block3gt(mask)
            x3_mask = x3_mask / (torch.max(x3_mask)+EFS)            

            mask = self.block4gt(x3_mask)
            mask = self.block5gt(mask)
            mask = self.block6gt(mask)
            mask = self.block7gt(mask)
            mask = self.block8gt(mask)
            mask = self.block9gt(mask)
            mask = self.block10gt(mask)
            mask = self.block11gt(mask)
            mask = self.block12gt(mask)
            mask = self.block13gt(mask)
            mask = self.block14gt(mask)
            mask = self.block15gt(mask)
            mask = self.block16gt(mask)
            mask = self.block17gt(mask)
            mask = self.block18gt(mask)
            x4_mask = self.block19gt(mask)
            x4_mask = x4_mask / (torch.max(x4_mask)+EFS)
            
            mask = self.block20gt(x4_mask)
            x5_mask = self.conv3gt(mask)
            x5_mask = x5_mask / (torch.max(x5_mask)+EFS)

            x6_mask = self.conv4gt(x5_mask)
            x6_mask = x6_mask / (torch.max(x6_mask)+EFS)

            mask = self.conv5gt(x6_mask)
            mask = mask / (torch.max(mask)+EFS)
        return [x1, x2, x3, x4, x5, x6, x7, x8, x9], [x1_multilayer_loss, x2_multilayer_loss, x3_multilayer_loss, x4_multilayer_loss, x5_multilayer_loss], x, low_level_feat, mask, low_level_feat_mask, self.uncertainty_output, self.uncertainty_gt


    def _init_imgxception_weight(self):
        for m in self.imgxceptionlist:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _init_maskxception_weight(self):
        for m in self.maskxceptionlist:
            if isinstance(m, nn.Conv2d):
                m.weight.data.fill_(1)
                if m.bias:
                    m.bias.data.zero_()


    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth')
        model_dict = {}
        state_dict = self.state_dict()

        for k, v in pretrain_dict.items():
            if k in state_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith('block11'):
                    model_dict[k] = v
                    model_dict[k.replace('block11', 'block12')] = v
                    model_dict[k.replace('block11', 'block13')] = v
                    model_dict[k.replace('block11', 'block14')] = v
                    model_dict[k.replace('block11', 'block15')] = v
                    model_dict[k.replace('block11', 'block16')] = v
                    model_dict[k.replace('block11', 'block17')] = v
                    model_dict[k.replace('block11', 'block18')] = v
                    model_dict[k.replace('block11', 'block19')] = v
                elif k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

