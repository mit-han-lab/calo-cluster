from functools import partial
import numpy as np
from sklearn.neighbors import NearestNeighbors

import numpy as np
import pytorch_lightning as pl


class MeanShift(pl.LightningModule):
    def __init__(self, *, bandwidth, seeds, cluster_all, max_iter, n_jobs):
        super().__init__()
        if type(seeds) is str:
            seeds = None
        self.meanshift = partial(mean_shift_cpu, bandwidth=bandwidth, seeds=seeds, cluster_all=cluster_all, max_iter=max_iter, n_jobs=n_jobs)

    def forward(self, embedding):
        embedding = embedding.cpu().numpy()
        _, labels = self.meanshift(embedding)
        return labels

def mean_shift_cpu(
    X,
    bandwidth,
    *,
    seeds=None,
    cluster_all=True,
    max_iter=300,
    n_jobs=None,
):
    if seeds is None:
        seeds = X
    
    stop_thresh = 1e-3 * bandwidth
    for _ in range(max_iter):
        dist2 = np.sum((seeds[:, None] - X[None, :]) ** 2, axis=2)
        masks = (dist2 <= bandwidth ** 2)

        seeds_ = seeds
        seeds = np.sum(X[None, ...] * masks[..., None], axis=1) / np.sum(masks, axis=1, keepdims=True)

        diff2 = np.sum((seeds - seeds_) ** 2, axis=-1)
        if (diff2 < stop_thresh ** 2).all():
            break

    sizes = np.sum(masks, axis=1)

    densities = {}
    for k in range(len(seeds)):
        if sizes[k]:
            densities[tuple(seeds[k])] = sizes[k]

    if not densities:
        raise ValueError(
            "No point was within bandwidth=%f of any seed." % bandwidth
        )

    densities = sorted(
        densities.items(),
        key=lambda tup: (tup[1], tup[0]),
        reverse=True,
    )
    seeds = np.array([tup[0] for tup in densities])

    dist2 = np.sum((seeds[:, None] - seeds[None, :]) ** 2, axis=2)

    mask = np.ones(len(seeds), dtype=bool)
    for k, seed in enumerate(seeds):
        if mask[k]:
            mask *= dist2[k] > bandwidth ** 2
            mask[k] = 1
    seeds = seeds[mask]
    
    nbrs = NearestNeighbors(n_neighbors=1, n_jobs=n_jobs).fit(seeds)
    labels = np.zeros(X.shape[0], dtype=int)
    distances, indices = nbrs.kneighbors(X)
    
    if cluster_all:
        labels = indices.flatten()
    else:
        labels.fill(-1)
        mask = distances.flatten() <= bandwidth
        labels[mask] = indices.flatten()[mask]

    return seeds, labels