import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from addict import Dict
from .necks.aspp_bcl import build_aspp
from .decoders.decoder_bcl import build_decoder
from .backbones import build_backbone
from addict import Dict

def sample_index( h, w, out_number): #在feature上sample

    indice_xy = []
    indice_x = []
    indice_y = []
    if h == 1 and w == 1:
        indice_xy.append([0, 0])
        indice_x.append(0)
        indice_y.append(0)
    else:
        step = int(w*h / out_number)
        end = int(w*h / out_number) * out_number
        index = 1
        while(index < end):
            if index % w != 0:
                index_x = index % w - 1
                index_y = int(index / w)
            elif index % w == 0 and index != 0:
                index_x = w - 1  
                index_y = index / w -1
            elif index == 1:
                index_x = 0
                index_y = 0
            
            index = index + step

            indice_xy.append([index_y, index_x])
            indice_x.append(index_x)
            indice_y.append(index_y)

    return indice_xy, indice_x, indice_y

def samplefor_strategyrf(data, sampleindex_y, sampleindex_x):
    sampled_data = data[:,:, sampleindex_y, sampleindex_x]
    return sampled_data


def cal_center_location_in_orimask(receptive_field_dict, patch_max_length, inputsize_w, inputsize_h, device):
    inputsize_w = inputsize_w - 2 * patch_max_length
    inputsize_h = inputsize_h - 2 * patch_max_length
    stride_x = receptive_field_dict.stride.x
    stride_y = receptive_field_dict.stride.y
    rfsize_w = receptive_field_dict.rfsize.w
    rfsize_h = receptive_field_dict.rfsize.h
    outputsize_w = receptive_field_dict.outputsize.w - 2 * patch_max_length #
    outputsize_h = receptive_field_dict.outputsize.h - 2 * patch_max_length
    center_location_in_orimask = torch.zeros(6, outputsize_w, outputsize_h).to(device)

    for x in range(outputsize_w):
        for y in range(outputsize_h):
            coor_x = (x + patch_max_length) * stride_x 
            coor_y = (y + patch_max_length) * stride_y 
            indice_x_min = int(coor_x - (rfsize_w - 1) / 2)
            indice_x_max = int(coor_x + (rfsize_w - 1) / 2)
            indice_y_min = int(coor_y - (rfsize_h - 1) / 2)
            indice_y_max = int(coor_y + (rfsize_h - 1) / 2)
            center_location_in_orimask[0,x,y] = indice_x_min
            center_location_in_orimask[1,x,y] = indice_x_max
            center_location_in_orimask[2,x,y] = indice_y_min
            center_location_in_orimask[3,x,y] = indice_y_max
            center_location_in_orimask[4,x,y] = coor_x
            center_location_in_orimask[5,x,y] = coor_y            

    return center_location_in_orimask


def cal_numberof_one(inputmask, patchsize, patchnumber, inputsize_w, inputsize_h, sampleindex, center_location_in_orimask, w, h, batch_size, device):
    inputmask = inputmask.squeeze(1)
    ratioofone = torch.zeros(batch_size, patchnumber , len(sampleindex)).to(device) 
    for j in range(inputmask.size()[0]): 

        k = 0
        for length in patchsize: 
            i = 0
            for y, x in sampleindex:
                y = int(y)
                x = int(x)
                [indice_x_min, indice_x_max, indice_y_min, indice_y_max, center_x, center_y] = center_location_in_orimask[:, y, x]
                
                square_input = inputmask[j, int(center_y-length/2):int(center_y+length/2), int(center_x-length/2):int(center_x+length/2)]
            
                if torch.numel(square_input) == 0:
                    ratioofone[j][k][i] = 0
                    pass
                numberofone = float(torch.sum(square_input) / torch.numel(square_input))
                ratioofone[j][k][i] = numberofone 
                i = i+1
         
            k = k + 1
    return ratioofone
