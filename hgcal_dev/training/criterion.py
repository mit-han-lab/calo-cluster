import random

import torch
from torch import nn
from torch import float32
from torch.nn import functional as F
from typing import List

def centroid_instance_loss(outputs, labels, subbatch_indices, weights, use_weights, normalize, delta_d, delta_v, batch_dim=0):
        # Add unity weights if none are provided.
        if weights is None or not use_weights:
            weights = torch.ones(labels.shape[0], dtype=float32, device=labels.device)

        # Normalize each output vector.
        if normalize:
            outputs = outputs / (torch.linalg.norm(outputs, axis=1) + 1e-8)[...,None]

        B = labels.shape[0]
        # Iterate over each event within the batch.
        if subbatch_indices is None:
            d = batch_dim
            subbatch = ((outputs[(slice(None),)*d+(i,)], labels[(slice(None),)*d+(i,)], weights[(slice(None),)*d+(i,)]) for i in torch.arange(B))
        else:
            unique_subbatch_indices = torch.unique(subbatch_indices)
            def get_subbatch():
                for subbatch_idx in unique_subbatch_indices:
                    subbatch_mask = subbatch_indices == subbatch_idx
                    subbatch_outputs = outputs[subbatch_mask]
                    subbatch_labels = labels[subbatch_mask]
                    subbatch_weights = weights[subbatch_mask]
                    yield (subbatch_outputs, subbatch_labels, subbatch_weights)
            subbatch = get_subbatch()

        loss = 0.0
        for subbatch_outputs, subbatch_labels, subbatch_weights in subbatch:
            unique_labels = torch.unique(subbatch_labels)
            mus = torch.zeros((unique_labels.shape[0], subbatch_outputs.shape[1]), device=subbatch_outputs.device)
            Ws = torch.zeros(unique_labels.shape[0])
            M = unique_labels.shape[0]

            # Find mean of each instance and calculate L_pull.
            if M > 1:
                L_pull = 0.0
                for m, label in enumerate(unique_labels):
                    mask = subbatch_labels == label
                    Nm = mask.sum()
                    mu = subbatch_outputs[mask].mean(axis=0)
                    W = subbatch_weights[mask].mean(axis=0)
                    Ws[m] = W
                    mus[m] = mu
                    L_pull += (subbatch_weights[mask] * F.relu(torch.norm(mu - subbatch_outputs[mask], p=1, dim=1) - delta_v)**2).sum() / (M * Nm)
                L_push = (F.relu(2 * delta_d - torch.norm(mus.unsqueeze(1) - mus, p=1, dim=2)).fill_diagonal_(0)**2).sum() / (M * (M - 1))
                loss += (L_pull + L_push) / B
        return loss

class CentroidInstanceLoss(nn.Module):
    def __init__(self, delta_v: float = 0.5, delta_d: float = 1.5, normalize: bool = True, use_weights: bool = False, method: str = None, ignore_label: int = None, batch_dim: int = 0) -> None:
        """ If method == 'all', make no distinction between semantic classes.
            If method == 'ignore', ignore any point with a semantic label equal to the given ignore_label.
            If method == 'separate', do the same as for 'ignore', but also calculate the loss for each semantic class separately.
        """
        super().__init__()
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.normalize = normalize
        self.use_weights = use_weights
        if method not in ['all', 'ignore', 'separate']:
            raise ValueError('invalid method!')
        self.method = method
        self.ignore_label = ignore_label
        self.batch_dim = batch_dim

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor, subbatch_indices: torch.Tensor, weights: torch.Tensor = None, semantic_labels: torch.Tensor = None):
        if self.method == 'all':
            loss = centroid_instance_loss(outputs, labels, subbatch_indices, weights, self.use_weights, self.normalize, self.delta_d, self.delta_v, self.batch_dim)
        elif self.method == 'ignore':
            mask = semantic_labels != self.ignore_label
            if subbatch_indices is not None:
                subbatch_indices = subbatch_indices[mask]
            if weights is not None:
                weights = weights[mask]
            loss = centroid_instance_loss(outputs[mask], labels[mask], subbatch_indices, weights, self.use_weights, self.normalize, self.delta_d, self.delta_v, self.batch_dim)
        elif self.method == 'separate':
            loss = 0.0
            unique_semantic_labels = torch.unique(semantic_labels)
            for semantic_label in unique_semantic_labels:
                if semantic_label == self.ignore_label:
                    continue
                mask = (semantic_labels == semantic_label)
                if subbatch_indices is not None:
                    subbatch_indices = subbatch_indices[mask]
                if weights is not None:
                    weights = weights[mask]
                loss += centroid_instance_loss(outputs[mask], labels[mask], subbatch_indices, weights, self.use_weights, self.normalize, self.delta_d, self.delta_v, self.batch_dim)
        return loss
                
def main():
    criterion = CentroidInstanceLoss(normalize=False)
    outputs = torch.arange(10, dtype=float32).reshape((5, 2))
    labels = torch.Tensor([2, 0, 2, 1, 2])
    subbatch_indices = torch.Tensor([0, 0, 0, 0, 0])
    print(criterion(outputs, labels, subbatch_indices))

if __name__ == "__main__":
    main()