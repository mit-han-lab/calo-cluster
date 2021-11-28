from functools import partial
from typing import Optional, Tuple
import numpy as np
from sklearn.neighbors import NearestNeighbors

import numpy as np
import torch
import pytorch_lightning as pl

class MeanShiftCuda(pl.LightningModule):
    def __init__(self, *, bandwidth, seeds, cluster_all, max_iter):
        super().__init__()
        if type(seeds) is str:
            seeds = None
        self.meanshift = partial(mean_shift_cuda, bandwidth=bandwidth, seeds=seeds, cluster_all=cluster_all, max_iter=max_iter)

    def forward(self, embedding):
        _, labels = self.meanshift(embedding)
        return labels


def mean_shift_cuda(
    X: torch.Tensor,
    *,
    bandwidth: float,
    seeds: Optional[torch.Tensor] = None,
    cluster_all: bool = True,
    max_iter: int = 300,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if seeds is None:
        seeds = X
    seeds = seeds.clone()

    mask = torch.ones(seeds.shape[0], dtype=torch.bool, device=seeds.device)
    for k in range(max_iter):
        dist2 = torch.sum((seeds[mask, None] - X[None, :]) ** 2, dim=2)
        nbrs = (dist2 <= bandwidth ** 2)

        seeds_ = seeds.clone()
        seeds[mask] = torch.sum(X[None, ...] * nbrs[..., None], axis=1) \
                    / torch.sum(nbrs, axis=1, keepdim=True)

        diff2 = torch.sum((seeds - seeds_) ** 2, axis=1)
        mask = (diff2 >= (bandwidth * 1e-3) ** 2)
        if not mask.any():
            break

    dist2 = torch.sum((seeds[:, None] - X[None, :]) ** 2, dim=2)
    sizes = torch.sum(dist2 <= bandwidth ** 2, axis=1)
    
    indices = (sizes > 0)
    seeds, sizes = seeds[indices], sizes[indices]
    
    if seeds.shape[0] == 0:
        raise ValueError(
            'No point was within bandwidth=%f of any seed.' % bandwidth
        )
    
    indices = torch.argsort(sizes, descending=True)
    seeds = seeds[indices]

    dist2 = torch.sum((seeds[:, None] - seeds[None, :]) ** 2, dim=2)
    ranks = torch.sum(torch.triu(dist2 > bandwidth ** 2, diagonal=1), dim=0)
    seeds = seeds[ranks == torch.arange(ranks.shape[0], device=ranks.device)]
    
    nbrs = NearestNeighbors(n_neighbors=1).fit(seeds.cpu().numpy())
    distances, indices = nbrs.kneighbors(X.cpu().numpy())
    
    if cluster_all:
        labels = torch.tensor(indices.flatten(), device=X.device)
    else:
        labels = torch.full(X.shape[0], -1, dtype=np.int, device=X.device)
        mask = (distances.flatten() <= bandwidth)
        labels[mask] = indices.flatten()[mask]
    
    return seeds, labels