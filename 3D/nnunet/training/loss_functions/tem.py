import torch.nn as nn
import torch.nn.functional as F
import torch 
import numpy as np

# x = torch.tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]))
# x = x.unsqueeze(0).unsqueeze(0).float()
# print(x.size())

# max_pool = nn.MaxPool2d(kernel_size=1, stride=2)
# y = max_pool(x)

# print(y)

def get_weight(weight):

    mask = torch.ones(weight.size(0), weight.size(1), weight.size(2), 1)
    print('应为B * 64 * 1', mask.size())  

    #print(mask.size(), weight.size())
    print(mask.size(), weight.unsqueeze(2).size())
    weight1 = torch.matmul(mask, weight.unsqueeze(2))
    print('应为B * 64 * 64', weight1.size(), weight.unsqueeze(2).size())

    weight2 = weight1 - weight.unsqueeze(2)
    weight2 = weight2.sum(dim=1)
    print('应为B * 64 * 64', weight2.size())

    #print('weight',weight)
    #print('weight1',weight1)
    #print('weight2', weight2)

    denominator = abs(weight2)
    numerator = 1 - denominator
    #print('应为B*64*64',denominator.size(),numerator.size())

    #print(mask, weight1, weight2)
    return numerator, denominator


weight = torch.ones((2,4,4,4,3))
weight = weight.view(weight.size(0), -1, weight.size(-1))
weight = weight.permute(0,2,1).contiguous()
print(weight.size())
n, d = get_weight(weight)
print(n.size(), d.size())
print(torch.unique(n), torch.unique(d))