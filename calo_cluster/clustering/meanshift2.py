from functools import partial
import numpy as np
from sklearn.neighbors import NearestNeighbors

import numpy as np

from calo_cluster.clustering.base_clusterer import BaseClusterer


class MeanShift2(BaseClusterer):
    def __init__(self, *, use_semantic, ignore_semantic_labels, bandwidth, seeds, cluster_all, max_iter, n_jobs):
        self.meanshift = partial(mean_shift_cpu, bandwidth=bandwidth, seeds=seeds, cluster_all=cluster_all, max_iter=max_iter, n_jobs=n_jobs)
        super().__init__(use_semantic, ignore_semantic_labels)

    def cluster(self, embedding, semantic_labels=None):
        """Clusters hits in event. If self.use_semantic, clusters only within each predicted semantic subset. 
           If self.ignore_semantic_labels, ignores hits with the given semantic labels."""
        if self.use_semantic:
            cluster_labels = np.full_like(semantic_labels, fill_value=-1)
            unique_semantic_labels = np.unique(semantic_labels)
            i = 0
            for l in unique_semantic_labels:
                if l in self.ignore_semantic_labels:
                    continue
                mask = (semantic_labels == l)
                self.meanshift(embedding[mask])
                cluster_labels[mask] = self.clusterer.labels_ + i
                i += np.unique(self.clusterer.labels_).shape[0]
        else:
            self.meanshift(embedding)
            cluster_labels = self.clusterer.labels_
        return cluster_labels

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