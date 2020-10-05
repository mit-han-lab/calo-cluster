from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

import cProfile
import torch
import pprofile
import random
import timeit
import tqdm
import logging

# sample triplets, with a weighted distribution if weights is specified.


def get_random_triplet_indices(labels, ref_labels=None, t_per_anchor=None, weights=None):
    assert weights is None or not torch.any(torch.isnan(weights))
    a_idx, p_idx, n_idx = [], [], []
    labels_device = labels.device
    ref_labels = labels if ref_labels is None else ref_labels
    ref_labels_is_labels = ref_labels is labels
    unique_labels = torch.unique(labels)
    unique_labels_ = unique_labels.view(-1, 1)
    p_masks = labels == unique_labels_
    l, ind = torch.nonzero(p_masks, as_tuple=True)
    for i, label in enumerate(unique_labels):
        # Get indices of positive samples for this label.
        p_inds = ind[l == label]
        n_a = p_inds.shape[0]
        if n_a < 2:
            continue
        k = p_inds.shape[0] if t_per_anchor is None else t_per_anchor
        p_inds_ = p_inds.expand((n_a, n_a))
        # Remove anchors from list of possible positive samples.
        p_inds_ = p_inds_[~torch.eye(n_a).bool()].view((n_a, n_a-1))
        # Get indices of indices of k random positive samples for each anchor.
        p_ = torch.randint(0, n_a-1, (n_a*k,))
        # Get indices of indices of corresponding anchors.
        a_ = torch.arange(n_a).view(-1, 1).repeat(1, k).view(n_a*k)
        p = p_inds_[a_, p_]
        a = p_inds[a_]

        # Get indices of negative samples for this label.
        n_inds = ind[l != label]
        if weights is not None:
            w = weights[:, n_inds][a]
            # Sample the negative indices according to the weights.
            n_ = torch.multinomial(w, k, replacement=True).flatten()
        else:
            # Sample the negative indices uniformly.
            n_ = torch.randint(0, n_inds.shape[0], (n_a*k,))
        n = n_inds[n_]
        a_idx.append(a)
        p_idx.append(p)
        n_idx.append(n)

    a_idx = torch.LongTensor(torch.cat(a_idx)).to(labels_device)
    p_idx = torch.LongTensor(torch.cat(p_idx)).to(labels_device)
    n_idx = torch.LongTensor(torch.cat(n_idx)).to(labels_device)
    return a_idx, p_idx, n_idx


def main():
    # profile()
    #new_time_test(100)
    #old_time_test(1)
    #validation_test(10)
    #completeness_test()
    one_hot_test()
    #old_one_hot_test()

def profile():
    embeddings = torch.rand(4000, 8)
    labels = torch.randint(0, 30, (4000,))
    profiler = pprofile.Profile()
    with profiler:
        lmu.get_random_triplet_indices(labels, t_per_anchor=1)
        #cProfile.runctx('get_random_triplet_indices(labels, t_per_anchor=1)', {'get_random_triplet_indices': get_random_triplet_indices}, {'labels': labels})

    profiler.dump_stats('old_profiler_stats.txt')

    profiler = pprofile.Profile()
    with profiler:
        get_random_triplet_indices(labels, t_per_anchor=1)
        #cProfile.runctx('get_random_triplet_indices(labels, t_per_anchor=1)', {'get_random_triplet_indices': get_random_triplet_indices}, {'labels': labels})

    profiler.dump_stats('new_profiler_stats.txt')


def new_triplets(n_min, n_max, l_min, l_max, t_per_anchor):
    n_samples = random.randint(n_min, n_max)
    n_labels = random.randint(l_min, l_max)
    labels = torch.randint(0, n_labels, (n_samples,))
    a, p, n = get_random_triplet_indices(labels, t_per_anchor=t_per_anchor)
    return a, p, n, labels


def old_triplets(n_min, n_max, l_min, l_max, t_per_anchor):
    n_samples = random.randint(n_min, n_max)
    n_labels = random.randint(l_min, l_max)
    labels = torch.randint(0, n_labels, (n_samples,))
    a, p, n = lmu.get_random_triplet_indices(labels, t_per_anchor=t_per_anchor)
    return a, p, n, labels


def new_time_test(n):
    t = timeit.timeit('new_triplets(1000, 10000, 10, 100, 10)',
                      setup='from __main__ import new_triplets', number=n)
    print(f'avg time = {t/n} s')


def old_time_test(n):
    t = timeit.timeit('old_triplets(1000, 10000, 10, 100, 100)',
                      setup='from __main__ import old_triplets', number=n)
    print(f'avg time = {t/n} s')


def validate_triplets(labels, a, p, n):
    l_a, l_p, l_n = labels[a], labels[p], labels[n]
    assert torch.all(a != p)
    assert torch.all(l_a == l_p)
    assert torch.all(l_p != l_n)


def validation_test(n):
    for _ in tqdm.tqdm(range(n)):
        a, p, n, labels = new_triplets(1000, 10000, 10, 100, 1)
        validate_triplets(labels, a, p, n)


def completeness_test():
    labels = torch.randint(0, 3, (5,))
    a_all, p_all, n_all = lmu.get_all_triplets_indices(labels)
    r_triplets = set()
    for _ in range(1000000):
        a_, p_, n_, = get_random_triplet_indices(labels, t_per_anchor=1)
        r_triplets = r_triplets.union(
            set((a, p, n) for a, p, n in zip(a_.numpy(), p_.numpy(), n_.numpy())))
    all_triplets = set((a, p, n) for a, p, n in zip(
        a_all.numpy(), p_all.numpy(), n_all.numpy()))
    assert len(all_triplets.difference(r_triplets)) == 0


def one_hot_test():
    labels = torch.randint(0, 3, (10,))
    w = torch.zeros((len(labels), len(labels)))
    for i, label in enumerate(labels):
        ind = torch.nonzero(labels != label)
        j = ind[torch.randint(0, len(ind), (1,))[0]]
        w[i, j] = 1.0
    a, p, n, = get_random_triplet_indices(labels, t_per_anchor=1, weights=w)
    assert torch.all(w[a, n] == 1.0)

def old_one_hot_test():
    labels = torch.randint(0, 3, (10,))
    w = torch.zeros((len(labels), len(labels)))
    for i, label in enumerate(labels):
        ind = torch.nonzero(labels != label)
        j = ind[torch.randint(0, len(ind), (1,))[0]]
        w[i, j] = 1.0
    breakpoint()
    a, p, n, = lmu.get_random_triplet_indices(labels, t_per_anchor=1, weights=w.numpy())
    breakpoint()
    assert torch.all(w[torch.arange(len(labels)), n] == 1.0)

if __name__ == '__main__':
    main()

