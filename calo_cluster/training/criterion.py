from typing import List, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss

try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse as ifilterfalse

# from DS-Net (https://github.com/hongfz16/DS-Net)
def offset_loss(pt_offsets, gt_offsets, valid):
    pt_diff = pt_offsets - gt_offsets   # (N, 3)
    pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)   # (N)
    valid = valid.view(-1).float()
    offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)
    return offset_norm_loss

def centroid_instance_loss(outputs, labels, subbatch_indices, normalize, delta_d, delta_v):
        # Normalize each output vector.
        if normalize:
            outputs = outputs / (torch.linalg.norm(outputs, axis=1) + 1e-8)[...,None]

        # Iterate over each event within the batch.
        unique_subbatch_indices = torch.unique(subbatch_indices)
        B = unique_subbatch_indices.shape[0]
        loss = 0.0
        for subbatch_idx in unique_subbatch_indices:
            subbatch_mask = subbatch_indices == subbatch_idx
            subbatch_outputs = outputs[subbatch_mask]
            subbatch_labels = labels[subbatch_mask]

            unique_labels = torch.unique(subbatch_labels)
            mus = torch.zeros((unique_labels.shape[0], subbatch_outputs.shape[1]), device=subbatch_outputs.device)
            M = unique_labels.shape[0]

            L_pull = 0.0
            for m, label in enumerate(unique_labels):
                mask = subbatch_labels == label
                Nm = mask.sum()
                mu = subbatch_outputs[mask].mean(axis=0)
                mus[m] = mu
                L_pull += (F.relu(torch.norm(mu - subbatch_outputs[mask], p=1, dim=1) - delta_v)**2).sum() / (M * Nm)
            if M > 1:
                dists = F.relu(2 * delta_d - torch.norm(mus.unsqueeze(1) - mus, p=1, dim=2))
                mask = torch.ones_like(dists).fill_diagonal_(0)
                L_push = ((dists*mask)**2).sum() / (M * (M - 1))
                loss += (L_pull + L_push) / B
            else:
                loss += L_pull / B
        return loss

class OffsetInstanceLoss(nn.Module):
    def __init__(self, valid_labels: List[int] = None, ignore_singleton: bool = False, weight: float = 1.0) -> None:
        super().__init__()
        if valid_labels is not None:
            self.valid_labels = torch.tensor(valid_labels)
        self.ignore_singleton = ignore_singleton
        self.weight = weight

    def forward(self, pt_offsets: torch.Tensor, gt_offsets: torch.Tensor, semantic_labels: torch.Tensor = None):
        if self.valid_labels is not None:
            if self.valid_labels.device != semantic_labels.device:
                self.valid_labels = self.valid_labels.to(semantic_labels.device)
            valid = (semantic_labels[..., None] == self.valid_labels).any(-1)
            
        else:
            valid = torch.ones_like(semantic_labels).type(torch.bool)
        if self.ignore_singleton:
            not_zero = (gt_offsets != 0).any(axis=1)
            valid = valid & not_zero
        loss = self.alpha * offset_loss(pt_offsets, gt_offsets, valid)
        return loss

class CentroidInstanceLoss(nn.Module):
    def __init__(self, delta_v: float = 0.5, delta_d: float = 1.5, normalize: bool = True, method: str = None, ignore_labels: List[int] = None, alpha: float = 1.0) -> None:
        """ If method == 'all', make no distinction between semantic classes.
            If method == 'ignore', ignore any point with a semantic label equal to the given ignore_labels.
            If method == 'separate', do the same as for 'ignore', but also calculate the loss for each semantic class separately.
        """
        super().__init__()
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.normalize = normalize
        if method not in ['all', 'ignore', 'separate']:
            raise ValueError('invalid method!')
        self.method = method
        if method in ['ignore', 'separate']:
            assert ignore_labels is not None
        self.ignore_labels = torch.tensor(ignore_labels)
        self.alpha = alpha

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor, subbatch_indices: torch.Tensor, weights: torch.Tensor = None, semantic_labels: torch.Tensor = None):
        if self.ignore_labels.device != outputs.device:
            self.ignore_labels = self.ignore_labels.to(outputs.device)
        if self.method == 'all':
            loss = centroid_instance_loss(outputs, labels, subbatch_indices, self.normalize, self.delta_d, self.delta_v)
        elif self.method == 'ignore':
            valid = ~(semantic_labels[..., None] == self.ignore_labels).any(-1)
            if subbatch_indices is not None:
                s_subbatch_indices = subbatch_indices[valid]
            else:
                s_subbatch_indices = None
            loss = centroid_instance_loss(outputs[valid], labels[valid], s_subbatch_indices, self.normalize, self.delta_d, self.delta_v)
        elif self.method == 'separate':
            loss = 0.0
            unique_semantic_labels = torch.unique(semantic_labels)
            for semantic_label in unique_semantic_labels:
                if semantic_label in self.ignore_labels:
                    continue
                mask = (semantic_labels == semantic_label)
                if subbatch_indices is not None:
                    s_subbatch_indices = subbatch_indices[mask]
                else:
                    s_subbatch_indices = None
                loss += centroid_instance_loss(outputs[mask], labels[mask], s_subbatch_indices, self.normalize, self.delta_d, self.delta_v)
        return self.alpha * loss



def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas,
                   labels,
                   classes='present',
                   per_image=False,
                   ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(
            lovasz_softmax_flat(
                *flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore),
                classes=classes) for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore),
                                   classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() >= 3:
        if probas.dim() == 3:
            # assumes output of a sigmoid layer
            B, H, W = probas.size()
            probas = probas.view(B, 1, H, W, 1)
        elif probas.dim() == 4:
            B, C, H, W = probas.size()
            probas = probas.view(B, C, H, W, 1)
        B, C, H, W, Z = probas.size()
        probas = probas.permute(0, 2, 3, 4,
                                1).contiguous().view(-1,
                                                     C)  # B * H * W, C = P, C
        labels = labels.view(-1)
        if ignore is None:
            return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, weight: float, classes: str, ignore_index: int):
        super().__init__()
        self.weight = weight
        self.classes = classes
        self.ignore_index = ignore_index

    def forward(self, outputs: torch.tensor, targets: torch.tensor):
        return self.weight * lovasz_softmax(F.softmax(outputs, 1),
                              targets.int(),
                              self.classes,
                              ignore=self.ignore_index)

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight: float, class_weights: Union[List[float], None], ignore_index: int) -> None:
        super().__init__()
        self.weight = weight
        if class_weights is not None:
            class_weights = torch.tensor(class_weights)
        self.class_weights = class_weights
        self.ignore_index = ignore_index

    def forward(self, outputs: torch.tensor, targets: torch.tensor):
        if self.class_weights.device != outputs.device:
            self.class_weights = self.class_weights.to(outputs.device)
        return self.weight * F.cross_entropy(outputs, targets, weight=self.class_weights, ignore_index=self.ignore_index)