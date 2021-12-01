
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
        if task == 'instance' or task == 'panoptic':
            self.offset_concat = OffsetConcat(cfg)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            # if hasattr(m, 'kernel') and m.kernel is not None:
            #     nn.init.constant_(m.kernel, 0.01)
            # if hasattr(m, 'weight') and m.weight is not None:
            #     nn.init.constant_(m.weight, 0.01)
            # if hasattr(m, 'bias') and m.bias is not None:
            #     nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
        # print(self.backbone.conv1.kernel)
        # print(self.offset_concat.offset[0].weight)
        

    def num_inf_or_nan(self, x):
        return (torch.isinf(x.F).sum(), torch.isnan(x.F).sum())

    def cluster(self, inputs, outputs):
        pred_offsets = outputs['pred_offsets']
        coordinates = inputs['coordinates_raw']
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
            logging.warning(f'{valid.sum()} points could not be clustered due to GPU memory constraint.')
            

        return pred_instance_labels

    def devoxelize(self, inputs, outputs):
        subbatch_idx = inputs['features'].C[..., -1]
        subbatch_im_idx = inputs['inverse_map'].C[..., -1]
        subbatch_idx = inputs['features'].C[..., -1]
        subbatch_im_idx = inputs['inverse_map'].C[..., -1]
        inputs_list = []
        outputs_list = []
        for j in torch.unique(subbatch_idx):
            mask = (subbatch_idx == j)
            im_mask = (subbatch_im_idx == j)
            inverse_map = inputs['inverse_map'].F[im_mask]
            inputs_j = {}
            outputs_j = {}
            for k,v in inputs.items():
                if k == 'inverse_map':
                    continue
                if '_raw' in k:
                    inputs_j[k] = v.F[im_mask]
                else:
                    inputs_j[k] = v.F[mask][inverse_map]
            for k,v in outputs.items():
                outputs_j[k] = v[mask][inverse_map]
            inputs_list.append(inputs_j)
            outputs_list.append(outputs_j)
        return inputs_list, outputs_list

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
        inputs_list, outputs_list = self.devoxelize(inputs, outputs)
        self.offset_concat(inputs_list, outputs_list)
        if not self.training and self.hparams.task in ('instance', 'panoptic'):
            for inputs_i, outputs_i in zip(inputs_list, outputs_list):
                outputs_i['pred_instance_labels'] = self.cluster(inputs_i, outputs_i)
                if self.hparams.task == 'panoptic':
                    outputs_i['pred_semantic_labels'] = self.merge_instance_semantic(outputs_i['pred_semantic_labels'], outputs_i['pred_instance_labels'])
        return outputs, inputs_list, outputs_list

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
        outputs, inputs_list, outputs_list = self(inputs)
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
            for inputs, outputs in zip(inputs_list, outputs_list):
                for metric in self.metrics.values():
                    metric(inputs, outputs)
        
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
        semantic_labels = []
        pred_offsets = []
        offsets = []
        for inputs, outputs in zip(inputs_list, outputs_list):
            semantic_labels.append(inputs['semantic_labels_raw'])
            pred_offsets.append(outputs['pred_offsets'])
            offsets.append(inputs['offsets_raw'])
        semantic_labels = torch.cat(semantic_labels)
        pred_offsets = torch.cat(pred_offsets)
        offsets = torch.cat(offsets)
        valid = torch.isin(semantic_labels, self.valid_semantic_labels_for_clustering)
        losses = {}
        for name, criterion in self.instance_criterion.items():
            loss = criterion(pred_offsets, offsets, valid)
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



class OffsetConcat(pl.LightningModule):
    def __init__(self, cfg: OmegaConf) -> None:
        super().__init__()
        self.hparams.update(cfg)
        cs = [int(self.hparams.model.cr * x) for x in self.hparams.model.cs]
        embed_dim = self.hparams.model.embed_dim
        self.offset = nn.Sequential(
            nn.Linear(cs[8] + embed_dim, cs[8], bias=True),
            nn.BatchNorm1d(cs[8]),
            nn.ReLU()
        )
        self.offset_linear = nn.Linear(cs[8], embed_dim, bias=True)
        
    def forward(self, inputs_list, outputs_list):
        for inputs, outputs in zip(inputs_list, outputs_list):
            x = outputs['pred_instance_features']
            coordinates = inputs['coordinates_raw']
            outputs['pred_offsets'] = self.offset_linear(self.offset(torch.cat([x, coordinates], dim=1)))
