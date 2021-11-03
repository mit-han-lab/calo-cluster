import time
from collections import OrderedDict
from typing import Callable, List

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
import torchsparse.nn.functional as spf
from calo_cluster.utils.comm import is_rank_zero
from omegaconf import OmegaConf
from torchsparse.tensor import PointTensor, SparseTensor

from .utils import *

from .spvcnn_mst import SPVCNN_sem, SPVCNN_embedder_head

__all__ = ['SPVCNN']
            
class SPVCNN(pl.LightningModule):
    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.hparams.update(cfg)
        if is_rank_zero():
            self.save_hyperparameters(cfg)

        #self.hparams.optimizer._target_ = 'calo_cluster.training.optimizers.adam_factory'
        #self.hparams.scheduler._target_ = 'calo_cluster.training.schedulers.one_cycle_lr_factory'
        self.optimizer_factory = hydra.utils.instantiate(
            self.hparams.optimizer)
        self.scheduler_factory = hydra.utils.instantiate(
            self.hparams.scheduler)

        assert self.hparams.task == 'panoptic'
        sem_model = SPVCNN_sem.load_from_checkpoint(cfg.model.sem_path)
        sem_model.freeze()
        self.backbone = sem_model.backbone
        self.classifier = sem_model.classifier

        cs = [int(self.hparams.model.cr * x) for x in self.hparams.model.cs]
        self.embedder = SPVCNN_embedder_head(cs, self.hparams.model.embed_dim) 

        self.embed_criterion = hydra.utils.instantiate(self.hparams.embed_criterion)

    def num_inf_or_nan(self, x):
        return (torch.isinf(x.F).sum(), torch.isnan(x.F).sum())

    def forward(self, x):
        # x: SparseTensor z: PointTensor
        self.backbone.eval()
        self.classifier.eval()
        with torch.no_grad():
            x0, y, z = self.backbone(x)
            c_out = self.classifier(x0, y, z)
        out = (c_out, self.embedder(x0, y, z))    
        return out

    def configure_optimizers(self):
        optimizer = self.optimizer_factory(self.parameters())
        if self.scheduler_factory is not None:
            scheduler = self.scheduler_factory(optimizer, self.num_training_steps())
            scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
            return [optimizer], [scheduler]
        else:
            return optimizer

    def step(self, batch, batch_idx, split):
        inputs = batch['features']
        targets = batch['labels'].F.long()
        outputs = self(inputs)
        subbatch_indices = inputs.C[..., -1]
        weights = batch.get('weights')
        offsets = batch['offsets'].F
        if type(weights) is SparseTensor:
            weights = weights.F
        else:
            weights = None
        sync_dist = (split != 'train')

        embed_loss = self.embed_criterion(outputs[1], offsets, semantic_labels=targets[:, 0])
        self.log(f'{split}_embed_loss', embed_loss, sync_dist=sync_dist)
        loss = embed_loss
        if type(embed_loss) is not float:
            ret = {'loss': loss, 'embed_loss': embed_loss.detach()}
        else:
            ret = {'loss': loss}
        self.log(f'{split}_loss', loss, sync_dist=sync_dist)
        return ret

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, split='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, split='val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, split='test')

    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)     

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        num_steps = (batches // effective_accum) * self.trainer.max_epochs 
        num_steps += batches * self.trainer.max_epochs - num_steps * effective_accum
        print(f'num steps = {num_steps}')
        return num_steps
