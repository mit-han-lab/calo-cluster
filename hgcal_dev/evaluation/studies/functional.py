import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


def iou_match(outputs, targets, weights=None, threshold=0.5, semantic=False, ignore_index=None, ignore_class_labels=None, match_highest=False):
    ''' xi: predicted cluster labels
        yi: true cluster labels
        weights: weight for each sample
        threshold: iou threshold (must be >= 0.5)
        match_highest: if true, match each true cluster to the pred cluster with the highest iou, rather than using a threshold.'''
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

        if ignore_class_labels is not None and k in ignore_class_labels:
            continue

        xik = xi * (xs == k)
        yik = yi * (ys == k)

        xmask = xik != 0
        ymask = yik != 0

        xik = xik - 1
        yik = yik - 1

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
        if match_highest:
            indices = np.zeros(ious.shape[0])
            ious_per_y = ious * (xyyinstances == yinstances[..., None]).astype(int)
            ious_per_y = ious_per_y[ious_per_y.any(axis=1)]
            indices[np.argmax(ious_per_y, axis=1)] = 1
            indices = indices.astype(bool)
        else:
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


def response(events, clusterer, match_highest):
    responses = defaultdict(list)
    energies = defaultdict(list)
    unmatched_true = defaultdict(lambda: np.zeros(len(events)))
    unmatched_pred = defaultdict(lambda: np.zeros(len(events)))
    for i, event in tqdm(enumerate(events)):
        xi = clusterer.cluster(event)
        yi = event.input_event[event.instance_label].values
        if event.weight_name:
            weights = event.input_event[event.weight_name].values
        else:
            weights = None
        if clusterer.use_semantic:
            xs = event.pred_class_labels
            ys = event.input_event[event.class_label].values
            outputs = (xs, xi)
            targets = (ys, yi)
        else:
            outputs = xi
            targets = yi
        matched_pred, matched_truth, *_ = iou_match(
            outputs, targets, weights=weights, ignore_class_labels=(clusterer.ignore_class_label,), semantic=clusterer.use_semantic, match_highest=match_highest)
        for k in matched_pred:
            if clusterer.use_semantic:
                yik = yi[ys == k]
                xik = xi[xs == k]
                weights_y = weights[ys == k]
                weights_x = weights[xs == k]
            else:
                yik = yi
                xik = xi
                weights_y = weights
                weights_x = weights
            all_truth = np.unique(yik)
            all_pred = np.unique(xik)
            yim_mask = yik == matched_truth[k][..., None]
            matched_truth_energies = (weights_y * yim_mask.astype(int)).sum(axis=1)

            xim_mask = xik == matched_pred[k][..., None]
            matched_pred_energies = (weights_x * xim_mask.astype(int)).sum(axis=1)
            response = matched_pred_energies / matched_truth_energies
            responses[k].append(response)
            energies[k].append(matched_truth_energies)

            unmatched_truth = np.setdiff1d(
                all_truth, matched_truth[k], assume_unique=True)
            yim_mask = yik == unmatched_truth[..., None]
            unmatched_truth_energies = (weights_y * yim_mask.astype(int)).sum(axis=1)

            n_unmatched_true = len(all_truth) - len(matched_truth[k])
            n_unmatched_pred = len(all_pred) - len(matched_pred[k])
            responses[k].append(np.zeros(n_unmatched_true))
            energies[k].append(unmatched_truth_energies)
            unmatched_true[k][i] = n_unmatched_true
            unmatched_pred[k][i] = n_unmatched_pred

    response = {k: np.concatenate(responses[k]) for k in responses}
    energy = {k: np.concatenate(energies[k]) for k in energies}

    cluster_dfs = {k: pd.DataFrame(
        {'energy response': response[k], 'energy': energy[k]}) for k in response}
    event_dfs = {k: pd.DataFrame({'n_unmatched_true_clusters': unmatched_true[k],
                             'n_unmatched_pred_clusters': unmatched_pred[k]}) for k in unmatched_true}
    return cluster_dfs, event_dfs


def make_bins(x, y, nbins, lo, hi):
    "find mean and std error of y within bins of x determined by nbins, lo and hi"
    means = np.zeros(nbins)
    errors = np.zeros(nbins)
    bin_edges = np.linspace(lo, hi, num=nbins)
    bin_indices = np.digitize(x, bin_edges) - 1
    for i in range(nbins):
        mask = (bin_indices == i)
        y_bin = y[mask]
        means[i] = np.mean(y_bin)
        errors[i] = np.std(y_bin) / (np.sum(mask))**0.5  # standard error
    return means, errors, bin_edges
