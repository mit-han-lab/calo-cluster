import torch
from torch.autograd import Function
import torch.nn as nn
import sys
from torch.utils.cpp_extension import load
import time
import os
pd = os.path.dirname(__file__)
load_list = [os.path.join(pd, '../interpolation', 'devox.cpp'), os.path.join(pd, '../interpolation', 'devox.cu')]
ti_devox = load(
    'ti_devox', load_list, verbose=True)



def calc_ti_weights(pc, idx_query, scale=1.0):
    # TBD: normalize the weights to a probability distribution. Note that some indices are "-1".
    with torch.no_grad():
        # don't want points to lie exactly on grid
        pc_grid = pc
        # don't use np.floor then convert to torch. numerical errors.
        if scale > 1.:
            pc_floor = torch.floor(pc / scale) * scale
        else:
            pc_floor = torch.floor(pc)
        pc_ceil = pc_floor + scale
        pc_gridx = pc_grid[:, 0].view(-1, 1)
        pc_gridy = pc_grid[:, 1].view(-1, 1)
        pc_gridz = pc_grid[:, 2].view(-1, 1)
        pc_floorx = pc_floor[:, 0].view(-1, 1)
        pc_floory = pc_floor[:, 1].view(-1, 1)
        pc_floorz = pc_floor[:, 2].view(-1, 1)
        pc_ceilx = pc_ceil[:, 0].view(-1, 1)
        pc_ceily = pc_ceil[:, 1].view(-1, 1)
        pc_ceilz = pc_ceil[:, 2].view(-1, 1)
        pc_floorx = pc_floorx.float()
        pc_floory = pc_floory.float()
        pc_floorz = pc_floorz.float()
        pc_ceilx = pc_ceilx.float()
        pc_ceily = pc_ceily.float()
        pc_ceilz = pc_ceilz.float()
        weight000 = (pc_ceilx-pc_gridx) * (pc_ceily-pc_gridy) * (pc_ceilz-pc_gridz)
        weight001 = (pc_ceilx-pc_gridx) * (pc_ceily-pc_gridy) * (pc_gridz-pc_floorz)
        weight010 = (pc_ceilx-pc_gridx) * (pc_gridy-pc_floory) * (pc_ceilz-pc_gridz)
        weight011 = (pc_ceilx-pc_gridx) * (pc_gridy-pc_floory) * (pc_gridz-pc_floorz)
        weight100 = (pc_gridx-pc_floorx) * (pc_ceily-pc_gridy) * (pc_ceilz-pc_gridz)
        weight101 = (pc_gridx-pc_floorx) * (pc_ceily-pc_gridy) * (pc_gridz-pc_floorz)
        weight110 = (pc_gridx-pc_floorx) * (pc_gridy-pc_floory) * (pc_ceilz-pc_gridz)
        weight111 = (pc_gridx-pc_floorx) * (pc_gridy-pc_floory) * (pc_gridz-pc_floorz)

        all_weights = torch.cat([weight000, weight001, weight010, weight011, weight100, weight101, weight110, weight111], 1).transpose(1,0).contiguous()
        if scale > 1:
            all_weights /= scale ** 3
        all_weights[idx_query==-1] = 0
        all_weights /= all_weights.sum(0) + 1e-8
    return all_weights



class DevoxelizationGPU(Function):
    @staticmethod
    def forward(ctx, feat, indices, weights):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        #torch.cuda.synchronize()
        #st = time.time()
        
        #out = torch.zeros(indices.shape[0], feat.shape[1], device=feat.device)
        out = ti_devox.forward(feat.contiguous(), indices.contiguous().int(), weights.contiguous())
        
        ctx.for_backwards = (indices.contiguous().int(), weights, feat.shape[0])
        
        return out
    

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        indices, weights, n = ctx.for_backwards
        
        grad_features = ti_devox.backward(
            grad_out.contiguous(), indices, weights, n
        )
        
        #return torch.zeros(n, grad_out.shape[1], device=grad_out.device), None, None
       
        return grad_features, None, None
        

devoxelize = DevoxelizationGPU.apply


