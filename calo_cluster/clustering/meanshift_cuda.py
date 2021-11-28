from functools import partial
from typing import Optional, Tuple
import numpy as np
from sklearn.neighbors import NearestNeighbors

import numpy as np
import torch
from calo_cluster.clustering.base_clusterer import BaseClusterer

class MeanShiftCuda(BaseClusterer):
    def __init__(self, *, use_semantic, valid_semantic_labels, bandwidth, seeds, cluster_all, max_iter):
        if type(seeds) is str:
            seeds = None
        self.meanshift = partial(mean_shift_cuda, bandwidth=bandwidth, seeds=seeds, cluster_all=cluster_all, max_iter=max_iter)
        super().__init__(use_semantic, valid_semantic_labels)

    def cluster(self, embedding, semantic_labels=None):
        """Clusters hits in event. If self.use_semantic, clusters only within each predicted semantic subset. 
           If self.valid_semantic_labels, ignores hits without the given semantic labels."""
        if self.use_semantic:
            cluster_labels = np.full(semantic_labels.shape[0], fill_value=-1)
            unique_semantic_labels = torch.unique(semantic_labels)
            i = 0
            for l in unique_semantic_labels:
                if l not in self.valid_semantic_labels:
                    continue
                mask = (semantic_labels == l)
                _, labels = self.meanshift(embedding[mask])
                labels += i
                cluster_labels[mask.cpu().numpy()] = labels
                i += np.unique(labels).shape[0]
        else:
            _, labels = self.meanshift(embedding)
            cluster_labels = labels
        return cluster_labels


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
        labels = indices.flatten()
    else:
        labels = torch.full(X.shape[0], -1, dtype=np.int, device=X.device)
        mask = (distances.flatten() <= bandwidth)
        labels[mask] = indices.flatten()[mask]
    
    return seeds, labels