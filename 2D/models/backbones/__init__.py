from .resnet import *
from .resnet_bcl import *
from .resnet_scl import *
from .resnet_att2 import *
from .resnet_bcl_att2 import *

from .xception import *
from .xception_bcl import *
from .xception_att import *
from .xception_att2 import *
from .xception_bcl_att import *
from .xception_bcl_att2 import *
from .xception_bcl_att_cbam import *
from .xception_bcl_att_senet import *
from .xception_bcl_decoder_rf import *


from .drn import *
from .mobilenet import *



__all__ = ['build_backbone']


def build_backbone(cfgs, backbone, output_stride, BatchNorm, pretrained=True):
    if backbone == 'resnet':
        return ResNet101(output_stride, BatchNorm, pretrained=pretrained)
    elif backbone == 'xception':
        return AlignedXception(output_stride, BatchNorm, pretrained=pretrained)
    elif backbone == 'drn':
        return drn_d_54(BatchNorm, pretrained=pretrained)
    elif backbone == 'mobilenet':
        return MobileNetV2(output_stride, BatchNorm, pretrained=pretrained)
    elif backbone == 'resnet_bcl':
        return ResNet101_bcl(output_stride, BatchNorm, pretrained=pretrained)
    elif backbone == 'resnet_att2':
        return ResNet101_att2(cfgs, output_stride, BatchNorm, pretrained=pretrained)
    elif backbone == 'xception_bcl':
        return AlignedXception_bcl(cfgs, output_stride, BatchNorm, pretrained=pretrained)
    elif backbone == 'xception_bcl_decoder_rf':
        return AlignedXception_bcl_decoder_rf(cfgs, output_stride, BatchNorm, pretrained=pretrained)
    elif backbone == 'resnet_scl':
        return ResNet101_scl(cfgs, output_stride, BatchNorm, pretrained=pretrained)
    elif backbone == 'xception_att':
        return AlignedXception_att(cfgs, output_stride, BatchNorm, pretrained=pretrained)
    elif backbone == 'xception_att2':
        return AlignedXception_att2(cfgs, output_stride, BatchNorm, pretrained=pretrained)
    elif backbone == 'xception_bcl_att':
        return AlignedXception_bcl_att(cfgs, output_stride, BatchNorm, pretrained=pretrained)
    elif backbone == 'xception_bcl_att2':
        return AlignedXception_bcl_att2(cfgs, output_stride, BatchNorm, pretrained=pretrained)
    elif backbone == 'xception_bcl_att_cbam':
        return AlignedXception_bcl_att_cbam(cfgs, output_stride, BatchNorm, pretrained=pretrained)
    elif backbone == 'xception_bcl_att_senet':
        return AlignedXception_bcl_att_senet(cfgs, output_stride, BatchNorm, pretrained=pretrained)
    else:
        raise NotImplementedError
