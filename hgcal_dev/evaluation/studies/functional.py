import numpy as np
import pandas as pd
from tqdm import tqdm



def iou_match(outputs, targets, weights=None, threshold=0.5, semantic=False, ignore_index=None, ignore_semantic_labels=None):
    ''' xi: predicted cluster labels
        yi: true cluster labels
        weights: weight for each sample
        threshold: iou threshold (must be >= 0.5)'''
    assert threshold >= 0.5

    if not semantic:
        xs = np.zeros(outputs.shape[0])
        ys = xs
        xi = outputs
        yi = targets
    else:
        xs, xi = outputs
        ys, yi = targets

    if weights is None:
        weights = np.ones_like(xi, dtype=np.float64)

    mask = (ys != ignore_index)
    xs, xi = xs[mask], xi[mask] + 1
    ys, yi = ys[mask], yi[mask] + 1
    weights = weights[mask]

    _xyxinstances = {}
    _xyyinstances = {}
    _ious = {}
    _xmatched = {}
    _ymatched = {}
    _xareas = {}
    _yareas = {}
    _xmapping = {}
    _ymapping = {}
    _intersections = {}
    for k in range(len(np.unique(np.concatenate([xs, ys])))):

        if ignore_semantic_labels is not None and k in ignore_semantic_labels:
            continue

        xik = xi * (xs == k)
        yik = yi * (ys == k)

        xmask = xik != 0
        ymask = yik != 0

        xlabels = xik[xmask]
        xweights = weights[xmask]
        xinstances = np.unique(xlabels)
        xmapping = {k: idx for idx, k in enumerate(xinstances)}
        xmatched = np.array([False] * xinstances.shape[0])
        xareas = (np.broadcast_to(xweights, (len(xinstances), len(xweights)))
                  * (xlabels == xinstances[..., None]).astype(int)).sum(axis=1)

        ylabels = yik[ymask]
        yweights = weights[ymask]
        yinstances = np.unique(ylabels)
        ymapping = {k: idx for idx, k in enumerate(yinstances)}
        ymatched = np.array([False] * yinstances.shape[0])
        yareas = (np.broadcast_to(yweights, (len(yinstances), len(yweights)))
                  * (ylabels == yinstances[..., None]).astype(int)).sum(axis=1)

        if len(yinstances) == 0 and len(xinstances) == 0:
            continue

        xymask = np.logical_and(xmask, ymask)
        xyweights = weights[xymask]
        xylabels = xik[xymask] + (2 ** 32) * yik[xymask]
        xyinstances = np.unique(xylabels)
        intersections = (
            xyweights * (xylabels == xyinstances[..., None]).astype(int)).sum(axis=1)
        xyxinstances, xyyinstances = xyinstances % (
            2 ** 32), xyinstances // (2 ** 32)

        xyxmask = xlabels == xyxinstances[..., None]
        xyxareas = (xweights * xyxmask.astype(int)).sum(axis=1)

        xyymask = ylabels == xyyinstances[..., None]
        xyyareas = (yweights * xyymask.astype(int)).sum(axis=1)

        unions = xyxareas + xyyareas - intersections
        ious = intersections / np.maximum(unions, 1e-15)
        ious[intersections == unions] = 1.0
        indices = ious > 0.5

        xmatched[[xmapping[k] for k in xyxinstances[indices]]] = True
        ymatched[[ymapping[k] for k in xyyinstances[indices]]] = True

        _xyxinstances[k] = xyxinstances[indices]
        _xyyinstances[k] = xyyinstances[indices]
        _ious[k] = ious[indices]
        _xmatched[k] = xmatched
        _ymatched[k] = ymatched
        _xareas[k] = xareas
        _yareas[k] = yareas
        _xmapping[k] = xmapping
        _ymapping[k] = ymapping
        _intersections[k] = intersections

    return _xyxinstances, _xyyinstances, _ious, _xmatched, _ymatched, _xareas, _yareas, _xmapping, _ymapping, _intersections


def resolution(events, clusterer):
    resolutions = []
    energies = []
    n_unmatched = 0
    unmatched_true = np.zeros(len(events))
    unmatched_pred = np.zeros(len(events))
    for i, event in tqdm(enumerate(events)):
        xi = clusterer.cluster(event)
        yi = event.input_event[event.instance_label].values
        if event.weight_name:
            weights = event.input_event[event.weight_name].values
        else:
            weights = None
        matched_pred, matched_truth, _ = iou_match(
            xi, yi, weights=weights, ignore_label=clusterer.ignore_label, use_semantic=clusterer.use_semantic)
        all_truth = np.unique(yi)
        all_pred = np.unique(xi)
        yim_mask = yi == matched_truth[..., None]
        matched_truth_energies = (weights * yim_mask.astype(int)).sum(axis=1)

        xim_mask = xi == matched_pred[..., None]
        matched_pred_energies = (weights * xim_mask.astype(int)).sum(axis=1)
        resolution = matched_pred_energies / matched_truth_energies
        resolutions.append(resolution)
        energies.append(matched_truth_energies)

        unmatched_truth = np.setdiff1d(
            all_truth, matched_truth, assume_unique=True)
        yim_mask = yi == unmatched_truth[..., None]
        unmatched_truth_energies = (weights * yim_mask.astype(int)).sum(axis=1)

        n_unmatched_true = len(all_truth) - len(matched_truth)
        n_unmatched_pred = len(all_pred) - len(matched_pred)
        resolutions.append(np.zeros(n_unmatched_true))
        energies.append(unmatched_truth_energies)
        unmatched_true[i] = n_unmatched_true
        unmatched_pred[i] = n_unmatched_pred

    resolution = np.concatenate(resolutions)
    energy = np.concatenate(energies)

    cluster_df = pd.DataFrame(
        {'energy resolution': resolution, 'energy': energy})
    event_df = pd.DataFrame({'n_unmatched_true_clusters': unmatched_true,
                             'n_unmatched_pred_clusters': unmatched_pred})
    return cluster_df, event_df


def make_bins(x, y, nbins, lo, hi):
    "find mean and std error of y within bins of x determined by nbins, lo and hi"
    means = np.zeros(nbins)
    errors = np.zeros(nbins)
    bin_edges = np.linspace(lo, hi, num=nbins)
    bin_indices = np.digitize(x, bin_edges) - 1
    for i in nbins:
        mask = (bin_indices == i)
        y_bin = y[mask]
        means[i] = np.mean(y_bin)
        errors[i] = np.std(y_bin) / (np.sum(mask))**0.5  # standard error
    return means, errors, bin_edges
