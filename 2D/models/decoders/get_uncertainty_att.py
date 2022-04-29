import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from addict import Dict
import yaml
from addict import Dict
from torch.autograd import Variable

def get_uncertainty_att(feature, ori_mask, conv):

    conv1 = conv
    feature1 = feature
    uncertainty_output = conv1(feature1)

    uncertainty_output = nn.Softmax(dim=1)(uncertainty_output)

    eps = 1e-20
    top = uncertainty_output * torch.log(uncertainty_output + eps)
    bottom = torch.log(torch.Tensor([uncertainty_output.size()[1]])).cuda()
    entropy = (- top / bottom).sum(dim=1)
    att = 1 - entropy

    fea = feature + feature * att.unsqueeze(1)     

    return fea, None, None
