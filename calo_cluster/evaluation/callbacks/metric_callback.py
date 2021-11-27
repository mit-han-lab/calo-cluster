import logging
from typing import Any, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

import hydra
from dataclasses import dataclass
from calo_cluster.evaluation.metrics.metric import Metric

@dataclass
class MetricCallback(Callback):
    metric: Metric

    def __post__init__(self):
        super().__init__()

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        features = batch['features']
        outputs = pl_module(features)
        inverse_map = batch['inverse_map'].F.type(torch.long)
        subbatch_idx = features.C[..., -1]
        subbatch_im_idx = batch['inverse_map'].C[..., -1]
        for j in torch.unique(subbatch_idx):
            mask = (subbatch_idx == j)
            im_mask = (subbatch_im_idx == j)
            subbatch = {}
            for k,v in batch.items():
                if k == 'inverse_map':
                    continue
                if '_raw' in k:
                    subbatch[k] = v.F[im_mask]
                else:
                    subbatch[k] = v.F[mask][inverse_map[im_mask]]
            if type(outputs) is dict:
                for k,v in outputs.items():
                    subbatch[k] = v[mask][inverse_map[im_mask]]
            else:
                subbatch['outputs'] = outputs[mask][inverse_map[im_mask]]
            self.metric.add_from_dict(subbatch)
    
    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.metric.save()