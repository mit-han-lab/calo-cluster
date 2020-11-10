import torch
import torch.nn as nn
import numpy as np


class SparseTensor:
    def __init__(self, feat, coords, cur_tensor_stride=1):
        self.F = feat
        self.C = coords
        self.s = cur_tensor_stride
        self.coord_maps = {}
        self.kernel_maps = {}

    
    def check(self):
        if self.s not in self.coord_maps:
            self.coord_maps[self.s] = self.C
    
    def to(self, device):
        assert type(self.F) == torch.Tensor
        assert type(self.C) == torch.Tensor
        self.F = self.F.to(device)
        self.C = self.C.to(device)
        return self
    
    def __add__(self, another_tensor):
        new_feature = self.F + another_tensor.F
        new_tensor = SparseTensor(new_feature, self.C, self.s)
        new_tensor.coord_maps = self.coord_maps
        new_tensor.kernel_maps = self.kernel_maps
        return new_tensor
    
    """
    def add_coord_map(self, coord_map):
        self.coord_maps[self.s] = coord_map
    
    def add_kernel_map(self, kernel_map, kernel_size, dilation):
        self.kernel_maps["k%d_s%d_d%d"%(kernel_size, self.s, dilation)] = kernel_map
    
    def add_coord_hash(self, coord_hash):
        self.coord_hashs[self.s] = coord_hash
    """
