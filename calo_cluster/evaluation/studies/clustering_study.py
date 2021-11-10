from typing import Any, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT


class PQCallback(Callback):
    def __init__(self, clusterer, pq):
        super().__init__()
        self.clusterer = clusterer
        self.pq = pq

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
            pred_labels = self.clusterer.cluster(embedding=embedding[mask][inverse_map[im_mask]].cpu().numpy(), semantic_labels=semantic_labels_)
            self.pq.add((semantic_labels_, pred_labels), (semantic_labels_, instance_labels_))

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print(f'pq = {self.pq.compute()}')