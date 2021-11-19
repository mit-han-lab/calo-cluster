import logging
from typing import Any, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

# @ray.remote
# def cluster(embedding, semantic_labels):
#     pass


class ResponseCallback(Callback):
    def __init__(self, clusterer, metric, path, use_nn, use_target, target_instance_label_name):
        super().__init__()
        self.clusterer = clusterer
        self.metric = metric
        #ncpus = 5
        #ray.init(num_cpus=ncpus)
        #self.clusterers = [MeanShift.remote(**clusterer_cfg) for _ in range(ncpus)]
        self.path = path
        self.use_nn = use_nn
        self.use_target = use_target
        self.target_instance_label_name = target_instance_label_name


    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        features = batch['features']
        coordinates = batch['coordinates'].F
        embedding = (coordinates + pl_module(features))
        semantic_labels = batch['semantic_labels'].F
        semantic_labels_mapped = batch['semantic_labels_mapped'].F
        instance_labels_mapped = batch['instance_labels_mapped'].F
        energy = batch['weights_mapped'].F
        inverse_map = batch['inverse_map'].F.type(torch.long)
        subbatch_idx = features.C[..., -1]
        subbatch_im_idx = batch['inverse_map'].C[..., -1]
        for j in torch.unique(subbatch_idx):
            mask = (subbatch_idx == j)
            im_mask = (subbatch_im_idx == j)
            semantic_labels_ = semantic_labels_mapped[im_mask].cpu().numpy()
            instance_labels_ = instance_labels_mapped[im_mask].cpu().numpy()
            if self.use_target:
                pred_labels = batch[f'{self.target_instance_label_name}_mapped'].F[im_mask].cpu().numpy()
            elif self.use_nn:
                pred_labels = self.clusterer.cluster(embedding=embedding[mask][inverse_map[im_mask]], semantic_labels=semantic_labels[mask][inverse_map[im_mask]])
            else:
                pred_labels = self.clusterer.cluster(embedding=coordinates[mask][inverse_map[im_mask]], semantic_labels=semantic_labels[mask][inverse_map[im_mask]])
            outputs = (semantic_labels_, pred_labels)
            energy_ = energy[im_mask].cpu().numpy()
            targets = (semantic_labels_, instance_labels_)
            self.metric.add(outputs, targets, energy_)
    
    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logging.info(f'saving response data to {self.path}')
        self.metric.save(self.path)
        return