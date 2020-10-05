import torch
from torch.autograd import Function
import torch.nn as nn
import sys
from torch.utils.cpp_extension import load
import os
pd = os.path.dirname(__file__)
load_list = [os.path.join(pd, '../convolution', 'convolution.cpp'), os.path.join(pd, '../convolution', 'convolution.cu')]
sparseconv = load(
    'sparseconv', load_list, verbose=True)

class SpConvolution(Function):
    @staticmethod
    def forward(ctx, features, kernel, neighbor_map, neighbor_offset, transpose=False):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        r"""
        Parameters
        ----------
        features : torch.FloatTensor
            (N, c_in) Features of the input point cloud.
        kernel : torch.FloatTensor
            (K, c_in, c_out) Kernel. with K to be kernel volume.
        neighbor_map: torch.IntTensor
            (K, N) K-volumetric neighborhood of each point.
            For entries with value -1, it means that this neighbor is inexistent.
        
        
        Returns
        -------
        torch.FloatTensor
            (N, c_out) The output tensor.
        """
        features = features.contiguous()
        kernel = kernel.contiguous()
        neighbor_map = neighbor_map.int().contiguous()
        neighbor_offset = neighbor_offset.int().contiguous()
        if not transpose:
            out = torch.zeros(neighbor_map[:,1].max()+1, kernel.size(-1), device=features.device)
        else:
            # tbd: ensure the original, upsampled size to be the same.
            out = torch.zeros(neighbor_map[:,0].max()+1, kernel.size(-1), device=features.device)
        sparseconv.forward(features, out, kernel, neighbor_map, neighbor_offset.cpu(), transpose)
        ctx.for_backwards = (features, kernel, neighbor_map, neighbor_offset.cpu(), transpose)
        return out
    

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        grad_out : torch.FloatTensor
            (N, c_out) tensor with gradients of ouputs

        Returns
        -------
        grad_features : torch.FloatTensor
            (N, c_in) tensor with gradients of features.

        grad_kernel: torch.FloatTensor
            (K, c_in, c_out) tensor with gradients of the kernel.

        None
        """

        features, kernel, neighbor_map, neighbor_offset, transpose = ctx.for_backwards
        K, c_in, c_out = kernel.size()
        N_in = features.size(0)
        N_out = grad_out.size(0)
        grad_features = torch.zeros(N_in, c_in, device=features.device)
        grad_kernel = torch.zeros(K, c_in, c_out, device=kernel.device)
        
        sparseconv.backward(
            features, grad_features, grad_out.contiguous(), 
            kernel, grad_kernel, neighbor_map, neighbor_offset, 
            transpose
        )
        return grad_features, grad_kernel, None, None, None


sparseconv_op = SpConvolution.apply