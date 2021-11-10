import time
from pathlib import Path

import numpy as np
import torch
from torchsparse.tensor import SparseTensor
from torchsparse.utils import sparse_quantize

from calo_cluster.clustering.mean_shift_cosine_gpu import MeanShiftCosine
from calo_cluster.models.spvcnn import SPVCNN


class TritonSPVCNN:
    def __init__(self, ckpt_path, bw=0.05, voxel_size=0.1, timing=False) -> None:
        self.model = SPVCNN.load_from_checkpoint(ckpt_path)
        self.clusterer = MeanShiftCosine(bandwidth=bw)
        self.voxel_size = voxel_size
        self.timing = timing
        if timing:
            self.timing_path = Path('timing.txt')
            if self.timing_path.exists():
                self.timing_path.unlink()

    def infer(self, feats: np.array):
        if self.timing:
            self.timing_file = self.timing_path.open('a+')
            start = time.time()

        features, inverse_map = self.make_tensors(feats)

        model = self.model
        model.cuda(0)
        model.eval()

        features = features.to(model.device)
        inverse_map = inverse_map.F.type(torch.long)

        if self.timing:
            nn_start = time.time()
        with torch.no_grad():
            embedding = model(features).cpu().numpy()
            embedding = embedding[inverse_map]

        if self.timing:
            nn_end = time.time()
            cluster_start = time.time()

        self.clusterer.fit(embedding)

        if self.timing:
            end = time.time()
            self.timing_file.write(
                f'{nn_end - nn_start} {end - cluster_start} {end - start}\n')
            self.timing_file.close()

        return self.clusterer.labels_

    def make_tensors(self, feats_: np.array):
        if self.timing:
            start = time.time()

        coords = feats_[:, :3]
        pc_ = np.round(coords / self.voxel_size)
        pc_ -= pc_.min(0, keepdims=1)
        _, inds, inverse_map = sparse_quantize(
            pc_, self.voxel_size, return_index=True, return_inverse=True)
        pc = torch.Tensor(pc_[inds])
        feat = torch.Tensor(feats_[inds])

        features = SparseTensor(feat, pc)
        inverse_map = SparseTensor(torch.Tensor(inverse_map), pc_)

        if self.timing:
            end = time.time()
            self.timing_file.write(f'{end - start} ')

        return features, inverse_map
