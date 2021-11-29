
from collections import defaultdict
import logging
from typing import List
import hydra
import pytorch_lightning as pl
from pytorch_lightning.utilities.warnings import rank_zero_warn
import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
from omegaconf import OmegaConf
from torchsparse.tensor import PointTensor

from calo_cluster.utils.comm import is_rank_zero
import time

from .utils import *
import math
from scipy import stats

class BaseLightningModule(pl.LightningModule):
    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.hparams.update(cfg)
        if is_rank_zero():
            self.save_hyperparameters(cfg)

        self.optimizer_factory = hydra.utils.instantiate(
            self.hparams.optimizer)
        self.scheduler_factory = hydra.utils.instantiate(
            self.hparams.scheduler)

        task = self.hparams.task
        assert task in ('instance', 'semantic', 'panoptic')
        if task == 'instance' or task == 'panoptic':
            self.instance_criterion = hydra.utils.instantiate(
                self.hparams.instance_criterion)
        if task == 'semantic' or task == 'panoptic':
            self.semantic_criterion = hydra.utils.instantiate(
                self.hparams.semantic_criterion)

        if task == 'instance' or task == 'panoptic':
            self.clusterer = hydra.utils.instantiate(self.hparams.clusterer)

        if 'metrics' in self.hparams:
            self.metrics = nn.ModuleDict(hydra.utils.instantiate(self.hparams.metrics))
        else:
            self.metrics = {}

        self.register_buffer('valid_semantic_labels_for_clustering', torch.tensor(self.hparams.dataset.valid_semantic_labels_for_clustering))
        self.backbone = hydra.utils.instantiate(self.hparams.model.backbone, cfg=cfg, _recursive_=False)

    def num_inf_or_nan(self, x):
        return (torch.isinf(x.F).sum(), torch.isnan(x.F).sum())

    def cluster(self, inputs, outputs):
        pred_offsets = outputs['pred_offsets']
        coordinates = inputs['coordinates']
        if self.hparams.task == 'panoptic':
            semantic_labels = outputs['pred_semantic_labels']
        else:
            semantic_labels = inputs['semantic_labels_raw']
        valid = torch.isin(semantic_labels, self.valid_semantic_labels_for_clustering)
        shifted_coordinates = coordinates[valid] + pred_offsets[valid]
        pred_instance_labels = torch.full_like(semantic_labels, fill_value=-1)
        try:
            pred_instance_labels[valid] = self.clusterer(shifted_coordinates)
        except:
            pass
            

        return pred_instance_labels

    def unbatch(self, inputs, outputs):
        subbatch_idx = inputs['features'].C[..., -1]
        subbatch_im_idx = inputs['inverse_map'].C[..., -1]
        inputs_list = []
        outputs_list = []
        for j in torch.unique(subbatch_idx):
            mask = (subbatch_idx == j)
            im_mask = (subbatch_im_idx == j)
            inputs_j = {}
            outputs_j = {}
            for k,v in inputs.items():
                if '_raw' in k or k == 'inverse_map':
                    inputs_j[k] = v.F[im_mask]
                else:
                    inputs_j[k] = v.F[mask]
            for k,v in outputs.items():
                outputs_j[k] = v[mask]
            inputs_list.append(inputs_j)
            outputs_list.append(outputs_j)
        return inputs_list, outputs_list

    def devoxelize(self, inputs_list, outputs_list):
        """If not training, devoxelize every input/output. For training, devoxelization is unnecessary, so do nothing."""
        if self.training:
            return {}, {}
        inputs_devox = []
        outputs_devox = []
        for inputs, outputs in zip(inputs_list, outputs_list):
            inverse_map = inputs['inverse_map']
            inputs_devox_j = {}
            outputs_devox_j = {}
            for k,v in inputs.items():
                if k == 'inverse_map':
                    continue
                if '_raw' in k:
                    inputs_devox_j[k] = v
                else:
                    inputs_devox_j[k] = v[inverse_map]
            for k,v in outputs.items():
                outputs_devox_j[k] = v[inverse_map]
            inputs_devox.append(inputs_devox_j)
            outputs_devox.append(outputs_devox_j)
        return inputs_devox, outputs_devox

    def merge_instance_semantic(self, sem, ins):
        if len(self.valid_semantic_labels_for_clustering) == 1:
            return sem
        sem = sem.clone()
        ins_ids = torch.unique(ins)
        for id in ins_ids:
            if id == -1: # id==-1 means stuff classes
                continue
            ind = (ins == id)
            sub_sem = sem[ind]
            mode_sem_id = torch.mode(sub_sem)[0]
            sem[ind] = mode_sem_id
        return sem

    def forward(self, inputs):
        outputs = self.backbone(inputs)
        inputs_list, outputs_list = self.unbatch(inputs, outputs)
        if not self.training and self.hparams.task in ('instance', 'panoptic'):
            for inputs_i, outputs_i in zip(inputs_list, outputs_list):
                outputs_i['pred_instance_labels'] = self.cluster(inputs_i, outputs_i)
                outputs_i['pred_semantic_labels'] = self.merge_instance_semantic(outputs_i['pred_semantic_labels'], outputs_i['pred_instance_labels'])
        inputs_devoxelized, outputs_devoxelized = self.devoxelize(inputs_list, outputs_list)
        return outputs, inputs_list, outputs_list, inputs_devoxelized, outputs_devoxelized

    def configure_optimizers(self):
        optimizer = self.optimizer_factory(self.parameters())
        if self.scheduler_factory is not None:
            scheduler = self.scheduler_factory(optimizer, self.num_training_steps)
            scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
            return [optimizer], [scheduler]
        else:
            return optimizer

    def validation_epoch_end(self, ret) -> None:
        for name, metric in self.metrics.items():
            result = metric.compute()
            if result.ndim == 0:
                self.log(f'val_{name}', result)
            metric.reset()

    def step(self, inputs, batch_idx, split):
        outputs, inputs_list, outputs_list, inputs_devoxelized, outputs_devoxelized = self(inputs)
        task = self.hparams.task
        if task == 'semantic':
            loss = self.semantic_step(inputs, outputs, split)
        elif task == 'instance':
            loss = self.instance_step(inputs_list, outputs_list, split)
        elif task == 'panoptic':
            loss = self.panoptic_step(inputs, outputs, inputs_list, outputs_list, split)
        else:
            raise RuntimeError("invalid task!")
        self.log(f'{split}_loss', loss)

        if split == 'val':
            for metric in self.metrics.values():
                metric(inputs_devoxelized, outputs_devoxelized)
        
        return loss

    def semantic_step(self, inputs, outputs, split):
        targets = inputs['semantic_labels'].F.long()

        losses = []
        for name, criterion in self.semantic_criterion.items():
            loss = criterion(outputs['pred_semantic_scores'], targets)
            self.log(f'{split}_{name}_loss', loss)
            losses.append(loss)
        loss = sum(losses)
        self.log(f'{split}_semantic_loss', loss)
        return loss

    def instance_step(self, inputs_list, outputs_list, split):
        losses = {}
        for name, criterion in self.instance_criterion.items():
            for inputs, outputs in zip(inputs_list, outputs_list):
                if self.hparams.task == 'panoptic':
                    semantic_labels = outputs['pred_semantic_labels']
                else:
                    semantic_labels = inputs['semantic_labels']
                valid = torch.isin(semantic_labels, self.valid_semantic_labels_for_clustering)
                loss = criterion(outputs['pred_offsets'], inputs['offsets'], valid)
                if name in losses:
                    losses[name] += loss
                else:
                    losses[name] = loss
            self.log(f'{split}_{name}_loss', losses[name])
        loss = sum(losses.values())
        self.log(f'{split}_instance_loss', loss)
        return loss

    def panoptic_step(self, inputs, outputs, inputs_list, outputs_list, split):
        semantic_loss = self.semantic_step(inputs, outputs, split)
        instance_loss = self.instance_step(inputs_list, outputs_list, split)
        return semantic_loss + instance_loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, split='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, split='val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, split='test')

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.num_training_batches != float('inf'):
            dataset_size = self.trainer.num_training_batches
        else:
            rank_zero_warn('Requesting dataloader...')
            dataset_size = len(self.trainer._data_connector._train_dataloader_source.dataloader())

        if isinstance(self.trainer.limit_train_batches, int):
            dataset_size = min(dataset_size, self.trainer.limit_train_batches)
        else:
            dataset_size = int(dataset_size * self.trainer.limit_train_batches)

        accelerator_connector = self.trainer._accelerator_connector
        if accelerator_connector.use_ddp2 or accelerator_connector.use_dp:
            effective_devices = 1
        else:
            effective_devices = self.trainer.devices

        effective_devices = effective_devices * self.trainer.num_nodes
        effective_batch_size = self.trainer.accumulate_grad_batches * effective_devices
        max_estimated_steps = math.ceil((dataset_size + effective_batch_size - 1) // effective_batch_size) * self.trainer.max_epochs

        max_estimated_steps = min(max_estimated_steps, self.trainer.max_steps) if self.trainer.max_steps != -1 else max_estimated_steps
        return max_estimated_steps