from sklearn.metrics import jaccard_score
import numpy as np

def match_instances(labels, pred_labels, return_maps=True):
    unique_labels = np.unique(labels)
    unique_pred_labels =  np.unique(pred_labels)
    ious = np.zeros((unique_labels.shape[0], unique_pred_labels.shape[0]))
    for i, truth_label in enumerate(unique_labels):
        for j, pred_label in enumerate(unique_pred_labels):
            _labels = labels.copy()
            _labels[labels == truth_label] = 1
            _labels[labels != truth_label] = 0
            _pred_labels = labels.copy()
            _pred_labels[pred_labels == pred_label] = 1
            _pred_labels[pred_labels != pred_label] = 0
            ious[i, j] = jaccard_score(_labels, _pred_labels, average='binary')
    if return_maps:
        truth_to_pred = {k: v for k,v in zip(unique_labels, ious.argmax(axis=1))}
        pred_to_truth = {v: k for k,v in truth_to_pred.items()}
        return truth_to_pred, pred_to_truth, ious.max(axis=1)
    else:
        return ious.max(axis=1)

def mIoU(labels, pred_labels):
    ious = match_instances(labels, pred_labels, return_maps=False)
    return ious.mean()