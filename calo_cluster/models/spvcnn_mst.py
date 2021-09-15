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

__all__ = ['SPVCNN']


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride), spnn.BatchNorm(outc),
            spnn.ReLU(True))

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        stride=stride,
                        transposed=True), spnn.BatchNorm(outc),
            spnn.ReLU(True))

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride), spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=1), spnn.BatchNorm(outc))

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1,
                            dilation=1, stride=stride),
                spnn.BatchNorm(outc)
        )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class SPVCNN_backbone(nn.Module):
    def __init__(self, cfg: OmegaConf):
        cs = [int(self.hparams.model.cr * x) for x in self.hparams.model.cs]

        self.stem = nn.Sequential(
            spnn.Conv3d(self.hparams.dataset.num_features,
                        cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True))

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[5] + cs[3], cs[5], ks=3, stride=1,
                              dilation=1),
                ResidualBlock(cs[5], cs[5], ks=3, stride=1, dilation=1),
            )
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[6] + cs[2], cs[6], ks=3, stride=1,
                              dilation=1),
                ResidualBlock(cs[6], cs[6], ks=3, stride=1, dilation=1),
            )
        ])

        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[7] + cs[1], cs[7], ks=3, stride=1,
                              dilation=1),
                ResidualBlock(cs[7], cs[7], ks=3, stride=1, dilation=1),
            )
        ])

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[4]),
                nn.BatchNorm1d(cs[4]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[4], cs[6]),
                nn.BatchNorm1d(cs[6]),
                nn.ReLU(True),
            )
        ])

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: SparseTensor z: PointTensor
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.hparams.model.pres,
                              self.hparams.model.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.stage1(x1)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        z1 = voxel_to_point(x4, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)

        y1 = point_to_voxel(x4, z1)
        y1.F = self.dropout(y1.F)
        y1 = self.up1[0](y1)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)
        z2 = voxel_to_point(y2, z1)
        z2.F = z2.F + self.point_transforms[1](z1.F)

        y3 = point_to_voxel(y2, z2)
        y3.F = self.dropout(y3.F)
        y3 = self.up3[0](y3)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)

        return y3, z2, x0


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

        task = self.hparams.criterion.task
        assert task in ('instance', 'semantic', 'panoptic')
        if task == 'instance' or task == 'panoptic':
            self.embed_criterion = hydra.utils.instantiate(
                self.hparams.embed_criterion)
        if task == 'semantic' or task == 'panoptic':
            self.semantic_criterion = hydra.utils.instantiate(
                self.hparams.semantic_criterion)

        


        if task == 'semantic' or task == 'panoptic':
            self.c_up4 = nn.ModuleList([
                BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlock(cs[8] + cs[0], cs[8], ks=3, stride=1,
                                dilation=1),
                    ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1),
                )
            ])
            self.c_point_transform = nn.Sequential(
                nn.Linear(cs[6], cs[8]),
                nn.BatchNorm1d(cs[8]),
                nn.ReLU(True),
            )
            self.c_lin = nn.Sequential(nn.Linear(cs[8],
                                                      self.hparams.dataset.num_classes))
        if task == 'instance' or task == 'panoptic':
            self.e_up4 = nn.ModuleList([
                BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlock(cs[8] + cs[0], cs[8], ks=3, stride=1,
                                dilation=1),
                    ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1),
                )
            ])
            self.e_point_transform = nn.Sequential(
                nn.Linear(cs[6], cs[8]),
                nn.BatchNorm1d(cs[8]),
                nn.ReLU(True),
            )
            self.e_lin = nn.Sequential(nn.Linear(cs[8],
                                                    self.hparams.model.embed_dim))

        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def classifier(self, x0, y, z):
        y4 = self.c_up4[0](y)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.c_up4[1](y4)
        z3 = voxel_to_point(y4, z)
        z3.F = z3.F + self.c_point_transform(z.F)
        return self.c_lin(z3.F)

    def embedder(self, y3, x0, z2):
        y4 = self.e_up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.e_up4[1](y4)
        z3 = voxel_to_point(y4, z2)
        z3.F = z3.F + self.e_point_transform(z2.F)
        return self.e_lin(z3.F)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def num_inf_or_nan(self, x):
        return (torch.isinf(x.F).sum(), torch.isnan(x.F).sum())

    def forward(self, x):
        # x: SparseTensor z: PointTensor
        x0, y, z, = self.backbone(x)


        task = self.hparams.criterion.task
        if task == 'semantic':
            out = self.classifier(x0, y, z)
        elif task == 'instance':
            out = self.embedder(x0, y, z)
        elif task == 'panoptic':
            out = (self.classifier(x0, y, z), self.embedder(x0, y, z))
        else:
            raise RuntimeError("invalid task!")
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
        weights = batch['weights']
        if type(weights) is SparseTensor:
            weights = weights.F
        else:
            weights = None
        sync_dist = (split != 'train')

        task = self.hparams.criterion.task
        if task == 'semantic':
            loss = self.semantic_criterion(outputs, targets)
            self.log(f'{split}_class_loss', loss, sync_dist=sync_dist)
            ret = {'loss': loss, 'class_loss': loss.detach()}
        elif task == 'instance':
            if self.hparams.criterion.requires_semantic:
                loss = self.embed_criterion(outputs, targets[:, 1], subbatch_indices, weights, semantic_labels=targets[:, 0])
            else:
                loss = self.embed_criterion(outputs, targets, subbatch_indices, weights)
            self.log(f'{split}_embed_loss', loss, sync_dist=sync_dist)
            
            ret = {'loss': loss, 'embed_loss': loss.detach()}
        elif task == 'panoptic':
            class_loss = self.semantic_criterion(outputs[0], targets[:, 0])
            self.log(f'{split}_class_loss', class_loss, sync_dist=sync_dist)
            embed_loss = self.embed_criterion(outputs[1], targets[:, 1], subbatch_indices, weights, semantic_labels=targets[:, 0])
            self.log(f'{split}_embed_loss', embed_loss, sync_dist=sync_dist)
            loss = class_loss + self.hparams.criterion.alpha * embed_loss
            if type(class_loss) is not float and type(embed_loss) is not float:
                ret = {'loss': loss, 'class_loss': class_loss.detach(), 'embed_loss': embed_loss.detach()}
            else:
                ret = {'loss': loss}
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