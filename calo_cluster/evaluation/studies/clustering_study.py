from typing import Any, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

import hydra

class PQCallback(Callback):
    def __init__(self, clusterer, pq, use_target, target_instance_label_name):
        super().__init__()
        self.clusterer = clusterer['clusterer']
        self.pq = pq
        self.use_target = use_target
        self.target_instance_label_name = target_instance_label_name

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        features = batch['features']
        coordinates = batch['coordinates'].F
        pred_offsets = pl_module(features)
        embedding = (coordinates + pred_offsets)
        semantic_labels = batch['semantic_labels_mapped'].F
        instance_labels = batch['instance_labels_mapped'].F
        inverse_map = batch['inverse_map'].F.type(torch.long)
        subbatch_idx = features.C[..., -1]
        subbatch_im_idx = batch['inverse_map'].C[..., -1]
        for j in torch.unique(subbatch_idx):
            mask = (subbatch_idx == j)
            im_mask = (subbatch_im_idx == j)
            semantic_labels_ = semantic_labels[im_mask].cpu().numpy()
            instance_labels_ = instance_labels[im_mask].cpu().numpy()
            if self.use_target:
                pred_labels = batch[f'{self.target_instance_label_name}_mapped'].F[im_mask].cpu().numpy()
            else:
                pred_labels = self.clusterer.cluster(embedding=embedding[mask][inverse_map[im_mask]], semantic_labels=batch['semantic_labels'].F[mask][inverse_map[im_mask]])
            self.pq.add((semantic_labels_, pred_labels), (semantic_labels_, instance_labels_))

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print(f'pq = {self.pq.compute()}')