import random

import torch
from torch import nn
from torch import float32
from torch.nn import functional as F
from typing import List

def centroid_instance_loss(outputs, labels, subbatch_indices, weights, use_weights, normalize, delta_d, delta_v, push_label: int = None, no_batch_indices: bool = False):
        # Add unity weights if none are provided.
        if weights is None or not use_weights:
            weights = torch.ones(labels.shape[0], dtype=float32, device=labels.device)

        # Normalize each output vector.
        if normalize:
            outputs = outputs / (torch.linalg.norm(outputs, axis=1) + 1e-8)[...,None]

        # Iterate over each event within the batch.
        if no_batch_indices:
            B = labels.shape[0]
            loss = 0.0
            for subbatch_idx in torch.arange(B):
                subbatch_outputs = outputs[subbatch_idx]
                subbatch_labels = labels[subbatch_idx]
                subbatch_weights = weights[subbatch_idx]

                unique_labels = torch.unique(subbatch_labels)
                mus = torch.zeros((unique_labels.shape[0], subbatch_outputs.shape[1]), device=subbatch_outputs.device)
                Ws = torch.zeros(unique_labels.shape[0])
                M = unique_labels.shape[0]

                # Find mean of each instance and calculate L_pull.
                if M > 1:
                    L_pull = 0.0
                    for m, label in enumerate(unique_labels):
                        if push_label is not None and label == push_label:
                            continue
                        mask = subbatch_labels == label
                        Nm = mask.sum()
                        mu = subbatch_outputs[mask].mean(axis=0)
                        W = subbatch_weights[mask].mean(axis=0)
                        Ws[m] = W
                        mus[m] = mu
                        L_pull += (subbatch_weights[mask] * F.relu(torch.norm(mu - subbatch_outputs[mask], p=1, dim=1) - delta_v)**2).sum() / (M * Nm)
                    L_push = (F.relu(2 * delta_d - torch.norm(mus.unsqueeze(1) - mus, p=1, dim=2)).fill_diagonal_(0)**2).sum() / (M * (M - 1))
                    loss += (L_pull + L_push) / B
        else:
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
                        if push_label is not None and label == push_label:
                            continue
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
    def __init__(self, delta_v: float = 0.5, delta_d: float = 1.5, separate_semantic_labels: bool = False, normalize: bool = True, use_weights: bool = False, ignore_semantic_label: int = None, push_label: int = -1) -> None:
        """ If separate_semantic_labels is set, apply the loss within each 'thing' semantic label separately. 
            This relies on the semantic segmentation being good and the final clustering being based on predicted semantic labels.

            If ignore_semantic_labels is not None, instances from these labels will not contribute to L_pull. 
            In this case, 'push_label' is used as a flag: any label equal to 'push_label' will be ignored.
        """
        super().__init__()
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.normalize = normalize
        self.use_weights = use_weights
        self.ignore_semantic_label = ignore_semantic_label
        self.separate_semantic_labels = separate_semantic_labels
        self.push_label = push_label

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor, subbatch_indices: torch.Tensor, weights: torch.Tensor = None, semantic_labels: torch.Tensor = None):
        if self.separate_semantic_labels:
            loss = 0.0
            unique_semantic_labels = torch.unique(semantic_labels)
            for semantic_label in unique_semantic_labels:
                if semantic_label == self.ignore_semantic_label:
                    continue
                mask = (semantic_labels == semantic_label)
                loss += centroid_instance_loss(outputs[mask], labels[mask], subbatch_indices[mask], weights[mask], self.use_weights, self.normalize, self.delta_d, self.delta_v)
        else:
            if self.ignore_semantic_label is not None:
                mask = semantic_labels == self.ignore_semantic_label
                labels = self.push_label * mask * labels
            else:
                self.push_label = None
            loss = centroid_instance_loss(outputs, labels, subbatch_indices, weights, self.use_weights, self.normalize, self.delta_d, self.delta_v, push_label=self.push_label)
        return loss
                
def main():
    criterion = CentroidInstanceLoss(normalize=False)
    outputs = torch.arange(10, dtype=float32).reshape((5, 2))
    labels = torch.Tensor([2, 0, 2, 1, 2])
    subbatch_indices = torch.Tensor([0, 0, 0, 0, 0])
    print(criterion(outputs, labels, subbatch_indices))

if __name__ == "__main__":
    main()