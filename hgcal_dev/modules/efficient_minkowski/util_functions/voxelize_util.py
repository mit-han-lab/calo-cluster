import torch
from torch.autograd import Function
import torch.nn as nn
import sys
from .hash_util import *
from torch.utils.cpp_extension import load
import os
pd = os.path.dirname(__file__)
load_list = [os.path.join(pd, '../others', 'insertion.cpp'), os.path.join(pd, '../others', 'insertion.cu')]
voxelize = load(
    'voxelize', load_list, verbose=True)

class VoxelizeGPU(Function):
    @staticmethod
    def forward(ctx, feat, idx, cnt):
        out = voxelize.insertion_forward(feat.float().contiguous(), idx.int().contiguous(), cnt)
        
        #out = torch.zeros(cnt.shape[0], feat.shape[1], device=feat.device)
        ctx.for_backwards = (idx.int().contiguous(), cnt, feat.shape[0])
        
        return out
    
    @staticmethod
    def backward(ctx, top_grad):
        idx, cnt, N = ctx.for_backwards
        #return torch.zeros(N, top_grad.shape[1], device=top_grad.device), None, None
        bottom_grad = voxelize.insertion_backward(top_grad.float().contiguous(), idx, cnt, N)
        return bottom_grad, None, None

    

voxelize_gpu = VoxelizeGPU.apply

