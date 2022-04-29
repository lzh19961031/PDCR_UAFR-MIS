#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import time
import itertools


class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss
        print('losss situation!!!', loss, weight_factors)

    def forward(self, x, y, cl_map, cl_map_resize):

        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i])
        

        patch_dragsaw_loss = cl_loss_3d(cl_map, y[0])
        patch_dragsaw_loss_multilayer = cl_loss_3d_multilayer(cl_map, cl_map_resize, y[0]) * 0.01
        patch_dragsaw_loss = patch_dragsaw_loss * 0.01 
        l = l + patch_dragsaw_loss + patch_dragsaw_loss_multilayer
        return l


def cl_loss_3d_multilayer(cl_map, cl_map_resize, gt):
    assert type(cl_map_resize) == list or type(cl_map_resize) == tuple
    patch_dragsaw_loss_multilayer = 0
    temperature = 0.5

    class_num = 3
    stride_list = [(8,8,8), (4,4,4)]
    rf_dict = {0:{'rf':(23,23,23), 'j':(4,4,4)}, 1:{'rf':(47,47,47), 'j':(8,8,8)}}

    for i in range(len(cl_map)):
        for j in range(i+1, len(cl_map)):
            feature1 = cl_map[i]
            max_pool1 = nn.MaxPool3d(kernel_size=1, stride=stride_list[i])
            fea_sampled1 = max_pool1(feature1)
            gt1 = gt
            foreground_ratio_matrix1 = foreground_ratio(feature1, fea_sampled1, gt1, stride=stride_list[i], rf_item=rf_dict[i], class_num=class_num)
            features_vector1 = fea_sampled1.view(fea_sampled1.size(0), fea_sampled1.size(1), -1)
            forefround_ratio_vector1 = foreground_ratio_matrix1.view(foreground_ratio_matrix1.size(0), -1, foreground_ratio_matrix1.size(-1))
            forefround_ratio_vector1 = forefround_ratio_vector1.permute(0,2,1).contiguous()

            feature2 = cl_map[j]
            max_pool2 = nn.MaxPool3d(kernel_size=1, stride=stride_list[j])
            fea_sampled2 = max_pool2(feature2)
            gt2 = gt
            foreground_ratio_matrix2 = foreground_ratio(feature2, fea_sampled2, gt2, stride=stride_list[j], rf_item=rf_dict[j], class_num=class_num)
            features_vector2 = fea_sampled2.view(fea_sampled2.size(0), fea_sampled2.size(1), -1)
            forefround_ratio_vector2 = foreground_ratio_matrix2.view(foreground_ratio_matrix2.size(0), -1, foreground_ratio_matrix2.size(-1))
            forefround_ratio_vector2 = forefround_ratio_vector2.permute(0,2,1).contiguous()

            feature1_ = cl_map_resize[i] 
            fea_sampled1_ = max_pool1(feature1_)
            features_vector1_ = fea_sampled1_.view(fea_sampled1_.size(0), fea_sampled1_.size(1), -1)

            feature2_ = cl_map_resize[j] 
            fea_sampled2_ = max_pool2(feature2_)
            features_vector2_ = fea_sampled2_.view(fea_sampled2_.size(0), fea_sampled2_.size(1), -1)

            similarity = nn.CosineSimilarity(dim=1)(features_vector1_.unsqueeze(3), features_vector2_.unsqueeze(2)) / temperature  
            similarity = similarity - torch.max(similarity)

            affinity_score, divergence_score = get_weight_multi_layer(forefround_ratio_vector1, forefround_ratio_vector2)
            affinity_score = affinity_score.cuda()
            divergence_score = divergence_score.cuda()

            positive = torch.exp(torch.mul(similarity, affinity_score))
            negative = torch.exp(torch.mul(similarity, divergence_score))
            negative = torch.sum(negative, 2).unsqueeze(2)
            negative = torch.repeat_interleave(negative, repeats=negative.size()[1], dim=2)

            logits = - torch.log(positive / negative).mean()
            patch_dragsaw_loss_multilayer = patch_dragsaw_loss_multilayer + logits
    patch_dragsaw_loss_multilayer /= len(cl_map_resize)
    return patch_dragsaw_loss_multilayer


def cl_loss_3d(cl_map, gt):
    assert type(cl_map) == list or type(cl_map) == tuple
    patch_dragsaw_loss = 0
    temperature = 0.5

    class_num = 3
    stride_list = [(8,8,8), (4,4,4)]
    rf_dict = {0:{'rf':(23,23,23), 'j':(4,4,4)}, 1:{'rf':(47,47,47), 'j':(8,8,8)}}

    for idx, feature in enumerate(cl_map):
        max_pool = nn.MaxPool3d(kernel_size=1, stride=stride_list[idx])
        fea_sampled = max_pool(feature)
        foreground_ratio_matrix = foreground_ratio(feature, fea_sampled, gt, stride=stride_list[idx], rf_item=rf_dict[idx], class_num=class_num)
        features_vector = fea_sampled.view(fea_sampled.size(0), fea_sampled.size(1), -1)
        forefround_ratio_vector = foreground_ratio_matrix.view(foreground_ratio_matrix.size(0), -1, foreground_ratio_matrix.size(-1))
        forefround_ratio_vector = forefround_ratio_vector.permute(0,2,1).contiguous()

        similarity = nn.CosineSimilarity(dim=1)(features_vector.unsqueeze(3), features_vector.unsqueeze(2)) / temperature  
        similarity = similarity - torch.max(similarity)

        affinity_score, divergence_score = get_weight(forefround_ratio_vector)
        affinity_score = affinity_score.cuda()
        divergence_score = divergence_score.cuda()

        positive = torch.exp(torch.mul(similarity, affinity_score))
        negative = torch.exp(torch.mul(similarity, divergence_score))
        negative = torch.sum(negative, 2).unsqueeze(2)
        negative = torch.repeat_interleave(negative, repeats=negative.size()[1], dim=2)

        logits = - torch.log(positive / negative).mean()
        patch_dragsaw_loss = patch_dragsaw_loss + logits

    patch_dragsaw_loss /= len(cl_map)

    return patch_dragsaw_loss


def foreground_ratio(feature, fea_sampled, gt, stride, rf_item, class_num):

    s_d, s_h, s_w = tuple(stride)

    b,c,d,h,w = fea_sampled.size()
    foreground_ratio_matrix = torch.zeros((b,d,h,w,class_num)) 
    jump = np.array(rf_item['j'])
    rf_d, rf_h, rf_w = tuple(((np.array(rf_item['rf'])-1)/2).astype(int))

    if feature.size(2) % s_d == 0:
        d = np.arange(feature.size(2)//s_d)*s_d
    else:
        d = np.arange(feature.size(2)//s_d+1)*s_d
    if feature.size(3) % s_h == 0:
        h = np.arange(feature.size(3)//s_h)*s_h
    else:
        h = np.arange(feature.size(3)//s_h+1)*s_h
    if feature.size(4) % s_w == 0:
        w = np.arange(feature.size(4)//s_w)*s_w
    else:
        w = np.arange(feature.size(4)//s_w+1)*s_w

    listOLists = [d, h, w] 

    sampled_coordinates = list(itertools.product(*listOLists)) 
    sampled_coordinates = np.array(sampled_coordinates)

    for cur_coord in sampled_coordinates:
        org_coord = cur_coord * jump
        for i in range(gt.size(0)):
            cropped_patch = gt[i, :,  
                            max(org_coord[0]-rf_d, 0): min(org_coord[0]+rf_d+1, gt.size(2)-1),
                            max(org_coord[1]-rf_h, 0): min(org_coord[1]+rf_h+1, gt.size(3)-1),
                            max(org_coord[2]-rf_w, 0): min(org_coord[2]+rf_w+1, gt.size(4)-1)]
            category, count = torch.unique(cropped_patch, return_counts=True)


            foreground_ratio = torch.zeros(class_num)
            if len(count) > 1:
                for idx, num in enumerate(count[1:]):
                    foreground_ratio[idx] = num / (rf_item['rf'][0]*rf_item['rf'][1]*rf_item['rf'][2])

            ad_coord = cur_coord/np.array(stride)
            foreground_ratio_matrix[i, int(ad_coord[0]), int(ad_coord[1]), int(ad_coord[2]), :] = foreground_ratio

    return foreground_ratio_matrix


def get_weight_multi_layer(weight1, weight2):

    mask = torch.ones(weight1.size(0), weight1.size(1), weight1.size(2), 1)
    weight1 = torch.matmul(weight1.unsqueeze(3), mask.permute(0,1,3,2))
    weight2 = torch.matmul(mask, weight2.unsqueeze(2))

    denominator = abs(weight1 - weight2).sum(dim=1)
    numerator = 1 - denominator

    return numerator, denominator


def get_weight(weight):

    mask = torch.ones(weight.size(0), weight.size(1), weight.size(2), 1)
    weight1 = torch.matmul(mask, weight.unsqueeze(2))
    weight2 = weight1 - weight.unsqueeze(3)
    weight2 = abs(weight2)

    denominator = weight2.sum(dim=1)
    numerator = 1 - denominator

    return numerator, denominator

