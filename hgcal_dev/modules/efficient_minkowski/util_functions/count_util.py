import torch
from torch.autograd import Function
import torch.nn as nn
import sys
from torch.utils.cpp_extension import load
import os

pd = os.path.dirname(__file__)
load_list = [os.path.join(pd, '../others', 'count.cpp'), os.path.join(pd, '../others', 'count.cu')]
count = load(
    'count', load_list, verbose=True)

class CountGPU(Function):
    @staticmethod
    def forward(ctx, idx, num):
        #return torch.ones(num).int()
        outs = count.forward(idx.int().contiguous(), num)
        return outs
    



count_gpu = CountGPU.apply