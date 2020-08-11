import torch
from torch.autograd import Function
import torch.nn as nn
import sys
from torch.utils.cpp_extension import load
import time
import os
pd = os.path.dirname(__file__)
load_list = [os.path.join(pd, '../others', 'query.cpp'), os.path.join(pd, '../hashmap', 'hashmap.cu')]
query = load(
    'query', load_list, verbose=True)


class SparseQuery(Function):
    @staticmethod
    def forward(ctx, hash_query, hash_target):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        if len(hash_query.size()) == 2:
            N, C = hash_query.size()
            outs = torch.zeros(N, C, device=hash_query.device).long()
        else:
            N = hash_query.size(0)
            C = 1
            outs = torch.zeros(N, device=hash_query.device).long()
                
        
        idx_target = torch.arange(len(hash_target)).to(hash_query.device).long()
        out, key_buf, val_buf, key = query.forward(hash_query.view(-1).contiguous(), hash_target.contiguous(), idx_target)
        if C > 1:
            out = out.view(-1, C)
        return out - 1
            
       
    



sparse_hash_query = SparseQuery.apply