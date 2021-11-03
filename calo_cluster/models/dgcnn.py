"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
Modified by 
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2020/3/9 9:32 PM
"""

import copy
import math
import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from calo_cluster.utils.comm import is_rank_zero
from omegaconf import OmegaConf
import hydra


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx




class DGCNN(pl.LightningModule):
    def __init__(self, cfg: OmegaConf):
        super(DGCNN, self).__init__()

        self.hparams = cfg
        if is_rank_zero():
            self.save_hyperparameters(cfg)
        
        self.optimizer_factory = hydra.utils.instantiate(
            self.hparams.optimizer)
        self.scheduler_factory = hydra.utils.instantiate(
            self.hparams.scheduler)

        task = self.hparams.task
        assert task in ('instance', 'semantic', 'panoptic')
        if task == 'instance' or task == 'panoptic':
            self.embed_criterion = hydra.utils.instantiate(
                self.hparams.criterion.embed)
        if task == 'semantic' or task == 'panoptic':
            self.semantic_criterion = hydra.utils.instantiate(
                self.hparams.criterion.semantic)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(8, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, 1024, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=self.hparams.model.dropout)

        if task == 'semantic' or task == 'panoptic':
            self.classifier = nn.Sequential(nn.Conv1d(256,
                                                      self.hparams.dataset.num_classes, kernel_size=1, bias=False))
        if task == 'instance' or task == 'panoptic':
            self.embedder = nn.Sequential(nn.Conv1d(256,
                                                    self.hparams.model.embed_dim, kernel_size=1, bias=False))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        batch_size = x.size(0)
        num_points = x.size(2)

        x = self.get_graph_feature(x, k=self.hparams.model.k)   # (batch_size, 3, num_points) -> (batch_size, 4*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 4*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = self.get_graph_feature(x1, k=self.hparams.model.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = self.get_graph_feature(x2, k=self.hparams.model.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)


        task = self.hparams.task
        if task == 'semantic':
            out = self.classifier(x)
        elif task == 'instance':
            out = self.embedder(x)
        elif task == 'panoptic':
            out = (self.classifier(x), self.embedder(x))
        else:
            raise RuntimeError("invalid task!")
        return out

    def configure_optimizers(self):
        optimizer = self.optimizer_factory(self.parameters())
        if self.scheduler_factory is not None:
            scheduler = self.scheduler_factory(optimizer, self.num_training_steps)
            scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
            return [optimizer], [scheduler]
        else:
            return optimizer

    def step(self, batch, batch_idx, split):
        inputs = batch['features']
        targets = batch['labels']
        outputs = self(inputs)
        #TODO: add weights
        weights = None
        sync_dist = (split != 'train')

        task = self.hparams.task
        if task == 'semantic':
            loss = self.semantic_criterion(outputs, targets)
            ret = {'loss': loss}
        elif task == 'instance':
            if self.hparams.requires_semantic:
                loss = self.embed_criterion(outputs, targets[:, 1], subbatch_indices=None, weights=weights, semantic_labels=targets[:, 0])
            else:
                loss = self.embed_criterion(outputs, targets, subbatch_indices=None, weights=weights)
            ret = {'loss': loss}
        elif task == 'panoptic':
            class_loss = self.semantic_criterion(outputs[0], targets[:, 0])
            self.log(f'{split}_class_loss', class_loss, sync_dist=sync_dist)
            embed_loss = self.embed_criterion(outputs[1], targets[:, 1], subbatch_indices=None, weights=weights, semantic_labels=targets[:, 0])
            self.log(f'{split}_embed_loss', embed_loss, sync_dist=sync_dist)
            loss = class_loss + embed_loss
            ret = {'loss': loss, 'class_loss': class_loss, 'embed_loss': embed_loss}
        else:
            raise RuntimeError("invalid task!")
        self.log(f'{split}_loss', loss, sync_dist=sync_dist)
        return ret

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, split='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, split='val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, split='test')

    @property
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

    def get_graph_feature(self, x, k=20, idx=None):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        if idx is None:
                idx = knn(x[:, :3], k=k)   # (batch_size, num_points, k)

        idx_base = torch.arange(0, batch_size, device=self.device).view(-1, 1, 1)*num_points

        idx = idx + idx_base

        idx = idx.view(-1)
    
        _, num_dims, _ = x.size()

        x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        feature = x.view(batch_size*num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims) 
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        
        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    
        return feature      # (batch_size, 2*num_dims, num_points, k)