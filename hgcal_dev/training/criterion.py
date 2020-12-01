import random

import torch
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.losses import TripletMarginLoss
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F


class JointLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, alpha: float = 1.0):
        self.alpha = alpha
        self.class_loss = CrossEntropyLoss(ignore_index=ignore_index)
        self.instance_loss = TripletMarginLoss()

    def forward(self, outputs, targets):
        embeddings, class_out = outputs
        instance_labels, class_labels = targets
        return self.alpha * self.instance_loss(embeddings, instance_labels) + self.class_loss(class_out, class_labels)


def triplet_margin_loss_factory(triplets_per_anchor, normalize_embeddings, p):
    criterion = TripletMarginLoss(
        triplets_per_anchor=triplets_per_anchor, distance=LpDistance(normalize_embeddings=normalize_embeddings, p=p))
    return criterion


class SHMarginLoss(nn.Module):
    def __init__(self, margin: float = 1) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = torch.mm(x, y.T)
        loss = (self.semi_hard_margin_loss(logits) +
                self.semi_hard_margin_loss(logits.T))
        return loss

    def semi_hard_margin_loss(self, logits: torch.Tensor) -> torch.Tensor:
        loss = 0
        size = logits.size(0)
        for k in range(size):
            anchor = logits[k][k]
            imposters, candidates = [], []
            for i in range(size):
                if i != k:
                    imposters.append(logits[k][i])
                if logits[k][i] < anchor:
                    candidates.append(logits[k][i])
            # sampling-based margin loss
            imposter = random.choice(imposters)
            diff = imposter - anchor + self.margin
            if (diff.data > 0).all():
                loss = loss + diff
            # semi-hard negative margin loss
            if candidates:
                imposter = max(candidates)
            else:
                imposter = random.choice(imposters)
            diff = imposter - anchor + self.margin
            if (diff.data > 0).all():
                loss = loss + diff
        return loss / size


class NTXentLoss(nn.Module):
    def __init__(self, tau: float = 0.1) -> None:
        super().__init__()
        self.tau = tau
        self.targets = dict()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = torch.mm(x, y.T) / self.tau
        size, device = logits.size(0), logits.device
        if (size, device) not in self.targets:
            self.targets[(size, device)] = torch.arange(size).to(device).long()
        targets = self.targets[(size, device)]
        loss = (F.cross_entropy(logits, targets) +
                F.cross_entropy(logits.T, targets))
        return loss
