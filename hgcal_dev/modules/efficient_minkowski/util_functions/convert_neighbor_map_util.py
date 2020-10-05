import os
import sys
import time

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.cpp_extension import load


pd = os.path.dirname(__file__)
load_list = [os.path.join(pd, '../others', 'convert_neighbor_map.cpp'),
             os.path.join(pd, '../others', 'convert_neighbor_map.cu')]
convert = load(
    'convert', load_list, verbose=True)

# convert the original neighbormap to the one without -1s.
# (using stream compaction algorithm)


class ConvertNeighborMap(Function):
    @staticmethod
    def forward(ctx, neighbor_map):
        idx_batch, idx_point = torch.where(neighbor_map != -1)
        map_converted = convert.convert_map_forward(
            neighbor_map.int(), idx_batch.int(), idx_point.int())
        nmap_offset = torch.sum(neighbor_map != -1, 1)
        return map_converted, nmap_offset


convert_neighbor_map_gpu = ConvertNeighborMap.apply
