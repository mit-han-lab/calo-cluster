import torch
import torch.nn as nn
import numpy as np


class PointTensor:
    def __init__(self, feat, coords, idx_query=None, weights=None):
        self.F = feat
        self.C = coords
        self.idx_query = idx_query if idx_query is not None else {}
        self.weights = weights if weights is not None else {}
        self.additional_features = {}
        self.additional_features['idx_query'] = {}
        self.additional_features['counts'] = {}

    
    
    def to(self, device):
        assert type(self.F) == torch.Tensor
        assert type(self.C) == torch.Tensor
        self.F = self.F.to(device)
        self.C = self.C.to(device)
        return self
    
    def __add__(self, another_tensor):
        new_feature = self.F + another_tensor.F
        new_tensor = PointTensor(new_feature, self.C, self.idx_query, self.weights)
        new_tensor.additional_features = self.additional_features
        return new_tensor

