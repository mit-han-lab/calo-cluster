import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .util_functions import *
from .sparse_tensor import *

import copy
import time


# TBD: kernel_size = 1 special case.
class MinkowskiConvolution(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, stride=1, dilation=1, transpose=False):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.ks = kernel_size
        self.k = self.ks ** 3
        self.s = stride
        self.d = dilation
        self.kernel = nn.Parameter(torch.zeros(self.k, inc, outc)) if self.k > 1 else nn.Parameter(torch.zeros(inc, outc))
        self.op = sparseconv_op
        self.t = transpose
        self.init_weight()
        
        if kernel_size == 1:
            assert not transpose
            
    def __repr__(self):
        if not self.t:
            return 'MinkowskiConvolution(in_channels=%d, out_channels=%d, kernel_size=%d, stride=%d, dilation=%d)'%(self.inc, self.outc, self.ks, self.s, self.d)
        else:
            return 'MinkowskiConvolutionTranspose(in_channels=%d, out_channels=%d, kernel_size=%d, stride=%d, dilation=%d)'%(self.inc, self.outc, self.ks, self.s, self.d)
    
    def init_weight(self):
        std = 1. / math.sqrt(self.outc if self.t else self.inc * self.k)
        self.kernel.data.uniform_(-std, std)
    
    
    def forward(self, inputs):
        # inputs: SparseTensor
        # outputs: SparseTensor
        
        features = inputs.F
        coords = inputs.C
        cur_stride = inputs.s
        
        if self.ks == 1:
            output_features = features.matmul(self.kernel)
            output_tensor = SparseTensor(output_features, coords, cur_stride)
            output_tensor.coord_maps = inputs.coord_maps
            output_tensor.kernel_maps = inputs.kernel_maps
            output_tensor.check()
        
        
        elif not self.t:
            kernel_map = inputs.kernel_maps.get('k%d_os%d_s%d_d%d'%(self.ks, cur_stride, self.s, self.d), None)
            
            if self.s > 1:
                # do downsample   
                kRegion = KernelRegion(kernel_size=self.ks, tensor_stride=cur_stride)
                kOffset = kRegion.get_kernel_offset().to(features.device)
                new_coords = downsample_gpu(coords, self.s * cur_stride)
                hash_query = kernel_hash_gpu(new_coords, kOffset)
                hash_target = hash_gpu(coords)
                idx_query = sparse_hash_query(hash_query, hash_target)
                idx_query = convert_neighbor_map_gpu(idx_query)
                output_features = self.op(features, self.kernel, idx_query[0], idx_query[1], self.t)
                output_tensor = SparseTensor(output_features, new_coords, cur_stride * self.s)
                output_tensor.coord_maps = copy.deepcopy(inputs.coord_maps)
                output_tensor.check()
                output_tensor.kernel_maps = copy.deepcopy(inputs.kernel_maps)
                output_tensor.kernel_maps['k%d_os%d_s%d_d%d'%(self.ks, cur_stride, self.s, self.d)] = idx_query 

            
            else:
                # submanifold sparseconv
                if kernel_map is None:
                    kRegion = KernelRegion(kernel_size=self.ks, tensor_stride=cur_stride)
                    try:
                        kOffset = kRegion.get_kernel_offset().to(features.device)
                    except:
                        print(features)
                        assert 0
                    hash_query = kernel_hash_gpu(coords, kOffset)
                    hash_target = hash_gpu(coords)
                    idx_query = sparse_hash_query(hash_query, hash_target)
                    idx_query = convert_neighbor_map_gpu(idx_query)
                    output_features = self.op(features, self.kernel, idx_query[0], idx_query[1], self.t)                
                    output_tensor = SparseTensor(output_features, coords, cur_stride)
                    output_tensor.coord_maps = inputs.coord_maps
                    output_tensor.check()
                    output_tensor.kernel_maps = copy.deepcopy(inputs.kernel_maps)
                    output_tensor.kernel_maps['k%d_os%d_s%d_d%d'%(self.ks, cur_stride, self.s, self.d)] = idx_query 
                else:                    
                    output_features = self.op(features, self.kernel, kernel_map[0], kernel_map[1], self.t)
                    output_tensor = SparseTensor(output_features, coords, cur_stride)
                    output_tensor.coord_maps = inputs.coord_maps
                    output_tensor.check()
                    output_tensor.kernel_maps = inputs.kernel_maps
                
                        
        
        else:
            # do upsample
            original_stride = int(cur_stride / self.s)
            kernel_map = inputs.kernel_maps.get('k%d_os%d_s%d_d%d'%(self.ks, original_stride, self.s, self.d), None)             
            output_features = self.op(features, self.kernel, kernel_map[0], kernel_map[1], self.t)
            output_tensor = SparseTensor(output_features, inputs.coord_maps[original_stride], original_stride)
            output_tensor.coord_maps = inputs.coord_maps
            output_tensor.kernel_maps = inputs.kernel_maps
        
        return output_tensor
        

class MinkowskiBatchNorm(nn.Module):
    def __init__(self, c, eps=1e-5, momentum=0.1):
        super().__init__()
        self.c = c
        self.eps = eps
        self.momentum = momentum
        self.bn = nn.BatchNorm1d(c, eps=eps, momentum=momentum)
        
    def __repr__(self):
        return 'MinkowskiBatchNorm(channels=%d)'%self.c
    
    def forward(self, inputs):
        features = inputs.F
        coords = inputs.C
        cur_stride = inputs.s
        output_features = self.bn(features)
        output_tensor = SparseTensor(output_features, coords, cur_stride)
        output_tensor.coord_maps = inputs.coord_maps
        output_tensor.kernel_maps = inputs.kernel_maps
        return output_tensor


class MinkowskiActivation(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.activation = None
        
    def forward(self, inputs):
        features = inputs.F
        coords = inputs.C
        cur_stride = inputs.s
        output_features = self.activation(features)
        output_tensor = SparseTensor(output_features, coords, cur_stride)
        output_tensor.coord_maps = inputs.coord_maps
        output_tensor.kernel_maps = inputs.kernel_maps
        return output_tensor
    

class MinkowskiReLU(MinkowskiActivation):
    def __init__(self, inplace=True):
        super().__init__()
        self.activation = nn.ReLU(True)
    
    def __repr__(self):
        return 'MinkowskiReLU(inplace=True)'
        
        
class MinkowskiConcatenation(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input_list):
        assert len(input_list) > 0
        inputs = input_list[0]
        features = inputs.F
        coords = inputs.C
        cur_stride = inputs.s
        output_tensor = SparseTensor(torch.cat([inputs.F for inputs in input_list], 1), coords, cur_stride)
        output_tensor.coord_maps = inputs.coord_maps
        output_tensor.kernel_maps = inputs.kernel_maps
        return output_tensor


class MinkowskiGlobalAveragePooling(nn.Module):
    def __init__(self):
        super().__init__()
    
    
    def forward(self, inputs):
        # outputs: batch_size X C
        batch_index = inputs.C[:, -1]
        max_index = torch.max(batch_index).item()
        outputs = []
        for i in range(max_index + 1):
            cur_inputs = torch.index_select(inputs.F, 0, torch.where(batch_index == i)[0])
            cur_outputs = cur_inputs.mean(0).unsqueeze(0)
            outputs.append(cur_outputs)
        outputs = torch.cat(outputs, 0)
        return outputs
    

class MinkowskiGlobalMaxPooling(nn.Module):
    def __init__(self):
        super().__init__()
    
    
    def forward(self, inputs):
        # outputs: batch_size X C
        batch_index = inputs.C[:, -1]
        max_index = torch.max(batch_index).item()
        outputs = []
        for i in range(max_index + 1):
            cur_inputs = torch.index_select(inputs.F, 0, torch.where(batch_index == i)[0])
            cur_outputs = cur_inputs.max(0)[0].unsqueeze(0)
            outputs.append(cur_outputs)
        outputs = torch.cat(outputs, 0)
        return outputs

