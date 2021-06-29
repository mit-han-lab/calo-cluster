import random

import torch
from torch import nn
from torch import float32
from torch.nn import functional as F

def centroid_instance_loss(outputs, labels, subbatch_indices, weights, use_weights, ignore_index, normalize, delta_d, delta_v):
        # Add unity weights if none are provided.
        if weights is None or not use_weights:
            weights = torch.ones(labels.shape[0], dtype=float32, device=labels.device)

        # Ignore points with invalid labels.
        mask = labels != ignore_index
        outputs = outputs[mask]
        labels = labels[mask]
        subbatch_indices = subbatch_indices[mask]
        weights = weights[mask]
        
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
            subbatch_weights = weights[subbatch_mask]

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
    def __init__(self, delta_v: float = 0.5, delta_d: float = 1.5, ignore_index: int = -1, normalize: bool = True, use_weights: bool = False, use_semantic: bool = False, semantic_ignore_index: int = 0) -> None:
        super().__init__()
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.ignore_index = ignore_index
        self.normalize = normalize
        self.use_weights = use_weights
        self.use_semantic = use_semantic
        self.semantic_ignore_index = semantic_ignore_index

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor, subbatch_indices: torch.Tensor, weights: torch.Tensor = None, semantic_labels: torch.Tensor = None):
        if self.use_semantic:
            loss = 0.0
            unique_semantic_labels = torch.unique(semantic_labels)
            for semantic_label in unique_semantic_labels:
                if (semantic_label == self.semantic_ignore_index) or semantic_label == self.ignore_index:
                    continue
                mask = (semantic_labels == semantic_label)
                loss += centroid_instance_loss(outputs[mask], labels[mask], subbatch_indices[mask], weights[mask], self.use_weights, self.ignore_index, self.normalize, self.delta_d, self.delta_v)
        else:
            loss = centroid_instance_loss(outputs, labels, subbatch_indices, weights, self.use_weights, self.ignore_index, self.normalize, self.delta_d, self.delta_v)
        return loss
                
def main():
    criterion = CentroidInstanceLoss(normalize=False)
    outputs = torch.arange(10, dtype=float32).reshape((5, 2))
    labels = torch.Tensor([2, 0, 2, 1, 2])
    subbatch_indices = torch.Tensor([0, 0, 0, 0, 0])
    print(criterion(outputs, labels, subbatch_indices))

if __name__ == "__main__":
    main()