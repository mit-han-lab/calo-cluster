from pytorch_lightning.metrics.functional.classification import average_precision, iou
import torch


def match_instances(pred, target):
    pass

def mAP(pred, target):
    classes = torch.unique(target)
    ap = torch.zeros(classes.shape[0])
    for i, c in enumerate(torch.unique(target)):
        ap[i] = average_precision(pred, target, pos_label=c)
    return torch.mean(ap)