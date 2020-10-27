import torch
from torch.autograd import Function
import torch.nn as nn
import sys
from .hash_util import *
from torch.utils.cpp_extension import load
import os
pd = os.path.dirname(__file__)
load_list = [os.path.join(pd, '../others', 'insertion.cpp'), os.path.join(pd, '../others', 'insertion.cu')]
downsample = load(
    'downsample', load_list, verbose=True)

class DownsampleGPU(Function):
    @staticmethod
    def forward(ctx, coords, ratio):
        '''
        Inputs:
        coords: torch.Int32 tensor, N x 4
        ratio: float, downsample ratio
        Outputs:
        coords_downsampled: torch.Int32 tensor, M x 4
        Algorithm: 
        Using torch.unique to get **inverse** indices
        Then use the insertion kernel.
        TBD:
        The insertion kernel w/o atomic op.
        '''
        coords_float = coords[:, :3].float()
        # following Minkowski engine
        coords_new = torch.floor(torch.floor(coords_float / ratio) * ratio).int()
        coords_new = torch.cat([coords_new, coords[:, 3].view(-1, 1)], 1)
        coords_new_hash =  hash_gpu(coords_new)
        uq, inv, cnt = torch.unique(coords_new_hash, return_inverse=True, return_counts=True)
        
        inv = inv.int()
        cnt = cnt.int()
        # rounding is necessary
        uq_coords = torch.round(downsample.insertion_forward(coords_new.float(), inv, cnt))
        uq_coords = uq_coords.int()
        
        
        # Notice: corrds_new_hash cannot be directly used
        return uq_coords#, coords_new_hash



downsample_gpu = DownsampleGPU.apply
