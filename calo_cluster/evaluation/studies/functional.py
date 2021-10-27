from collections import defaultdict

import numpy as np
import pandas as pd
from calo_cluster.evaluation.metrics.instance import PanopticQuality, iou_match
from tqdm.auto import tqdm


def cluster_width(events, clusterer, match_highest, num_classes):
    for i, event in tqdm(enumerate(events)):
        xi = clusterer.cluster(event)
        yi = event.input_event[event.instance_label].values
        if event.weight_name:
            weights = event.input_event[event.weight_name].values
        else:
            weights = None
        if clusterer.use_semantic:
            xs = event.pred_semantic_labels
            ys = event.input_event[event.semantic_label].values
            for k in np.unique(xs):
                pass
            outputs = (xs, xi)
            targets = (ys, yi)
        else:
            outputs = xi
            targets = yi
        
        matched_pred, matched_truth, *_ = iou_match(
            outputs, targets, weights=weights, ignore_semantic_labels=(clusterer.ignore_semantic_label,), semantic=clusterer.use_semantic, match_highest=match_highest, num_classes=num_classes)
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
            yim_mask = yik == matched_truth[k][..., None]
            xim_mask = xik == matched_pred[k][..., None]

def response(events, clusterer, match_highest, num_classes):
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
            xs = event.pred_semantic_labels
            ys = event.input_event[event.semantic_label].values
            outputs = (xs, xi)
            targets = (ys, yi)
        else:
            outputs = xi
            targets = yi
        matched_pred, matched_truth, *_ = iou_match(
            outputs, targets, weights=weights, ignore_semantic_labels=(clusterer.ignore_semantic_label,), semantic=clusterer.use_semantic, match_highest=match_highest, num_classes=num_classes)
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


def make_bins(x, y, bin_edges, use_standard_error: bool = True, use_median: bool = False):
    "find mean and std error of y within bins of x determined by bin_edges"
    nbins = len(bin_edges)
    means = np.zeros(nbins)
    errors = np.zeros(nbins)
    bin_indices = np.digitize(x, bin_edges) - 1
    for i in range(nbins):
        mask = (bin_indices == i)
        y_bin = y[mask]
        if use_median:
            means[i] = np.median(y_bin)
            errors[i] = np.abs(np.percentile(y_bin, 75) - np.percentile(y_bin, 25))
        else:
            means[i] = np.mean(y_bin)
            if use_standard_error:
                errors[i] = np.std(y_bin) / (np.sum(mask))**0.5  # standard error
            else:
                errors[i] = np.std(y_bin)
    return means, errors

def pq(event, use_weights, clusterer, task, ignore_semantic_labels):
    pred_instance_labels = clusterer.cluster(event)
    if task == 'panoptic':
        pq_metric = PanopticQuality(num_classes=event.num_classes, ignore_index=-1, ignore_semantic_labels=ignore_semantic_labels)

        outputs = (event.pred_semantic_labels, pred_instance_labels)
        targets = (event.input_event[event.semantic_label].values,
                event.input_event[event.instance_label].values)
    elif task == 'instance':
        pq_metric = PanopticQuality(semantic=False, ignore_index=-1)

        outputs = pred_instance_labels
        targets = event.input_event[event.instance_label].values

    if use_weights:
        if event.weight_name is None:
            raise RuntimeError('No weight name given!')
        weights = event.input_event[event.weight_name].values
    else:
        weights = None
    pq_metric.add(outputs, targets, weights=weights)

    pq = pq_metric.compute()
    return pq

def num_clusters(event, pred_instance_labels):
    pass
