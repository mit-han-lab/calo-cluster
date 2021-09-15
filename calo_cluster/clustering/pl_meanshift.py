# Author Mengyang Zhao <Mengyang.Zhao@tufts.edu>

# Based on: Conrad Lee <conradlee@gmail.com>
#           Alexandre Gramfort <alexandre.gramfort@inria.fr>
#           Gael Varoquaux <gael.varoquaux@normalesup.org>
#           Martino Sorbaro <martino.sorbaro@ed.ac.uk>  

import math
import operator
from random import shuffle

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.neighbors import NearestNeighbors
from torch import exp, sqrt

# seeds number intital
SEED_NUM = 128
L = 2
H = 8

class MeanShift(pl.LightningModule):
    def __init__(self, *, use_semantic=False, ignore_semantic_labels=None, bandwidth):
        super().__init__()
        self.use_semantic = use_semantic
        self.ignore_semantic_labels = ignore_semantic_labels
        self.bandwidth = bandwidth

    def forward(self, embedding, semantic_labels=None):
        """Clusters hits in event. If self.use_semantic, clusters only within each predicted semantic subset. 
           If self.ignore_semantic_labels, ignores hits with the given semantic labels."""
        if self.use_semantic:
            cluster_labels = np.full_like(semantic_labels, fill_value=-1)
            unique_semantic_labels = np.unique(semantic_labels)
            for l in unique_semantic_labels:
                if l in self.ignore_semantic_labels:
                    continue
                mask = (semantic_labels == l)
                self.cluster_centers_, self.labels_ = self.mean_shift_cosine(embedding[mask])
                cluster_labels[mask] = self.clusterer.labels_
        else:
            self.clusterer.fit(embedding)
            cluster_labels = self.clusterer.labels_
        return cluster_labels

    def mean_shift_cosine(self, X):
        if self.bandwidth <= 0:
            raise ValueError("bandwidth needs to be greater than zero or None,\
                got %f" % self.bandwidth)
        seeds = gpu_seed_generator(X)

        # adjusted=False
        n_samples = X.shape[0]
        center_intensity_dict = {}
        nbrs = NearestNeighbors(radius=self.bandwidth, metric='cosine').fit(X)
        #NearestNeighbors(radius=bandwidth, n_jobs=n_jobs, metric='cosine').radius_neighbors()

        global SEED_NUM
        while True:
            labels, number = self.meanshift_torch(X, seeds, 0.1)  # gpu calculation
            for i in range(len(number)):
                if number[i] is not None:
                    # find out cluster
                    center_intensity_dict[tuple(labels[i])] = number[i]

            if not center_intensity_dict:
                # nothing near seeds
                raise ValueError("No point was within bandwidth=%f of any seed."
                                    " Try a different seeding strategy \
                                or increase the bandwidth."
                                    % self.bandwidth)

            # POST PROCESSING: remove near duplicate points
            # If the distance between two kernels is less than the bandwidth,
            # then we have to remove one because it is a duplicate. Remove the
            # one with fewer points.

            sorted_by_intensity = sorted(center_intensity_dict.items(),
                                            key=lambda tup: (tup[1], tup[0]),
                                            reverse=True)
            sorted_centers = np.array([tup[0] for tup in sorted_by_intensity])
            unique = np.ones(len(sorted_centers), dtype=np.bool)
            nbrs = NearestNeighbors(
                radius=self.bandwidth, metric='cosine').fit(sorted_centers)
            for i, center in enumerate(sorted_centers):
                if unique[i]:
                    neighbor_idxs = nbrs.radius_neighbors([center],
                                                            return_distance=False)[0]
                    unique[neighbor_idxs] = 0
                    unique[i] = 1  # leave the current point as unique
            cluster_centers = sorted_centers[unique]

            # assign labels
            nbrs = NearestNeighbors(
                n_neighbors=1, metric='cosine').fit(cluster_centers)
            labels = np.zeros(n_samples, dtype=np.int)
            distances, idxs = nbrs.kneighbors(X)
            labels = idxs.flatten()

            bg_num = np.sum(labels == 0)
            r = 1-bg_num/labels.size
            # seed number adjust
            dict_len = len(cluster_centers)  # cluster number

            N = get_N(0.95, r, dict_len)

            if L*N <= SEED_NUM:  # safety area
                # SEED_NUM -= 200#test
                if H*N <= SEED_NUM:
                    SEED_NUM -= N  # seeds are too much, adjust

                break
            else:
                seeds = gpu_seed_adjust(X)  # seeds are too few, adjust

        return cluster_centers, labels




def gpu_seed_generator(codes):

    seed_indizes = list(range(codes.shape[0]))
    shuffle(seed_indizes)
    seed_indizes = seed_indizes[:SEED_NUM]
    seeds = codes[seed_indizes]

    return seeds


def gpu_seed_adjust(codes):
    global SEED_NUM
    SEED_NUM *= 2

    return gpu_seed_generator(codes)


def get_N(P, r, I):

    # There is no foreground instances
    if r < 0.1:
        return 32  # Allocated some seeds at least

    lnp = math.log(P, math.e)
    num = math.log(1-math.e**(lnp/I), math.e)
    den = math.log(1-r/I, math.e)
    result = num/den

    if result < 32:
        result = 32  # Allocated some seeds at least
    elif result > 256:
        result = 256  # Our GPU memory's max limitation, you can higher it.

    return int(result)

# Author Mengyang Zhao <Mengyang.Zhao@tufts.edu>

def cos_batch(a, b):
    num = a@b.T
    denom = torch.norm(a, dim=1).reshape(-1, 1) * torch.norm(b, dim=1)
    return num / denom


def get_weight(sim, bandwidth):

    thr = 1-bandwidth
    max = torch.tensor(1.0).double()
    min = torch.tensor(0.0).double()
    dis = torch.where(sim > thr, max, min)

    return dis


def gaussian(dist, bandwidth):
    return exp(-0.5 * ((dist / bandwidth))**2) / (bandwidth * math.sqrt(2 * math.pi))


def meanshift_torch(data, seed, bandwidth, max_iter=300):

    stop_thresh = 1e-3 * bandwidth
    iter = 0

    X = torch.from_numpy(np.copy(data)).double()
    S = torch.from_numpy(np.copy(seed)).double()
    B = torch.tensor(bandwidth).double()

    while True:
        weight = get_weight(cos_batch(S, X), B)

        num = (weight[:, :, None] * X).sum(dim=1)
        S_old = S
        S = num / weight.sum(1)[:, None]
        iter += 1

        if (torch.norm(S - S_old, dim=1).mean() < stop_thresh or iter == max_iter):
            break

    p_num = []
    for line in weight:
        p_num.append(line[line == 1].size()[0])

    my_mean = S.cpu().numpy()

    return my_mean, p_num
