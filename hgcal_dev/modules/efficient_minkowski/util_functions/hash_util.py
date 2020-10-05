import torch
from torch.autograd import Function
import torch.nn as nn
import sys
from torch.utils.cpp_extension import load
import os
pd = os.path.dirname(__file__)
load_list = [os.path.join(pd, '../hash', 'hash.cpp'), os.path.join(pd, '../hash', 'hash.cu')]
hashing = load(
    'hashing', load_list, verbose=True)

class HashGPU(Function):
    @staticmethod
    def forward(ctx, idx):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        #cumulated_batch_index = torch.cumsum(torch.bincount(batch_index), dim=0)
        return hashing.forward(idx.int().contiguous())
    

class KernelHashGPU(Function):
    @staticmethod
    def forward(ctx, idx, koffset):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        #cumulated_batch_index = torch.cumsum(torch.bincount(batch_index), dim=0)
        return hashing.kernel_forward(idx.int().contiguous(), koffset.int().contiguous())

    



hash_gpu = HashGPU.apply
kernel_hash_gpu = KernelHashGPU.apply