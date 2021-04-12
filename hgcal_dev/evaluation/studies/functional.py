import numpy as np
from tqdm import tqdm
import pandas as pd

def iou_match(xi, yi, weights=None, threshold=0.5):
    ''' xi: predicted cluster labels
        yi: true cluster labels
        weights: weight for each sample
        threshold: iou threshold (must be >= 0.5)'''
    assert threshold >= 0.5
    if weights is None:
        weights = np.ones_like(xi, dtype=np.float64)

    xylabels = xi + (2 ** 32) * yi
    xyinstances = np.unique(xylabels)

    xymask = xylabels == xyinstances[..., None]
    intersections = (weights * xymask.astype(int)).sum(axis=1)

    xyxinstances, xyyinstances = xyinstances % (
        2 ** 32), xyinstances // (2 ** 32)

    xyxmask = xi == xyxinstances[..., None]
    xyxareas = (weights * xyxmask.astype(int)).sum(axis=1)

    xyymask = yi == xyyinstances[..., None]
    xyyareas = (weights * xyymask.astype(int)).sum(axis=1)

    unions = xyxareas + xyyareas - intersections
    ious = intersections / unions
    indices = ious > threshold
    return xyxinstances[indices], xyyinstances[indices], ious[indices]

def resolution(events):
    resolutions = []
    energies = []
    n_unmatched = 0
    for event in tqdm(events):
        xi = event.pred_instance_labels
        yi = event.input_event[event.instance_label].values
        if event.weight_name:
            weights = event.input_event[event.weight_name].values
        else:
            weights = None
        matched_pred, matched_truth, _ = iou_match(
            xi, yi, weights=weights)
        all_truth = np.unique(yi)
        yim_mask = yi == matched_truth[..., None]
        matched_truth_energies = (weights * yim_mask.astype(int)).sum(axis=1)

        xim_mask = xi == matched_pred[..., None]
        matched_pred_energies = (weights * xim_mask.astype(int)).sum(axis=1)
        resolution = matched_pred_energies / matched_truth_energies
        resolutions.append(resolution)
        energies.append(matched_truth_energies)

        unmatched_truth = np.setdiff1d(all_truth, matched_truth, assume_unique=True)
        yim_mask = yi == unmatched_truth[..., None]
        unmatched_truth_energies = (weights * yim_mask.astype(int)).sum(axis=1)

        n_unmatched = len(all_truth) - len(matched_truth)
        resolutions.append(np.zeros(n_unmatched))
        energies.append(unmatched_truth_energies)
        
    resolution = np.concatenate(resolutions)
    energy = np.concatenate(energies)

    df = pd.DataFrame({'energy resolution': resolution, 'energy': energy})
    return df