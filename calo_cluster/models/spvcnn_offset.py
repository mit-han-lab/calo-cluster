
from collections import defaultdict
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

__all__ = ['SPVCNNOffset']


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


class SPVCNNBackbone(pl.LightningModule):
    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.hparams.update(cfg)

        task = self.hparams.task
        assert task in ('instance', 'semantic', 'panoptic')

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
        self.dropout = nn.Dropout(0.3, True)

    def classifier(self, y3, x0, z2):
        y4 = self.c_up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.c_up4[1](y4)
        z3 = voxel_to_point(y4, z2)
        z3.F = z3.F + self.c_point_transform(z2.F)
        return self.c_lin(z3.F)

    def embedder(self, y3, x0, z2):
        y4 = self.e_up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.e_up4[1](y4)
        z3 = voxel_to_point(y4, z2)
        z3.F = z3.F + self.e_point_transform(z2.F)
        out = self.e_lin(z3.F)
        if 'tanh_scale' in self.hparams.model and self.hparams.model.tanh_scale is not None:
            out = torch.tanh(out) * self.hparams.model.tanh_scale
        return out

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def num_inf_or_nan(self, x):
        return (torch.isinf(x.F).sum(), torch.isnan(x.F).sum())

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

        task = self.hparams.task
        out = {}
        if task == 'semantic' or task == 'panoptic':
            out['pred_semantic_scores'] = self.classifier(y3, x0, z2)
            out['pred_semantic_labels'] = out['pred_semantic_scores'].argmax(dim=1)
        if task == 'instance' or task == 'panoptic':
            out['pred_offsets'] = self.embedder(y3, x0, z2)
        return out

class SPVCNNOffset(pl.LightningModule):
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

        self.backbone = SPVCNNBackbone(cfg)

    def num_inf_or_nan(self, x):
        return (torch.isinf(x.F).sum(), torch.isnan(x.F).sum())

    def cluster(self, inputs, outputs):
        pred_offsets = outputs['pred_offsets']
        coordinates = inputs['coordinates']
        if self.hparams.task == 'panoptic':
            semantic_labels = outputs['pred_semantic_labels']
        else:
            semantic_labels = inputs['semantic_labels_raw']
        pred_instance_labels = []
        for i in len(pred_offsets):
            pred_instance_labels_i = torch.full_like(semantic_labels[i], fill_value=-1)
            valid = torch.isin(semantic_labels[i], self.hparams.dataset.valid_semantic_labels_for_clustering)
            shifted_coordinates = coordinates[i][valid] + pred_offsets[i][valid]
            pred_instance_labels_i[valid] = self.clusterer(shifted_coordinates)
            pred_instance_labels.append(pred_instance_labels_i)

        return pred_instance_labels

    def unbatch(self, inputs, outputs, production=False):
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


    def forward(self, inputs):
        outputs = self.backbone(inputs['features'])
        inputs_list, outputs_list = self.unbatch(inputs, outputs)
        if not self.training and self.hparams.task in ('instance', 'panoptic'):
            outputs_list['pred_instance_labels'] = self.cluster(inputs_list, outputs_list)
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
                loss = criterion(outputs['pred_offsets'], inputs['offsets'], semantic_labels=outputs['semantic_labels'].long())
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
        max_estimated_steps = math.ceil(dataset_size // effective_batch_size) * self.trainer.max_epochs

        max_estimated_steps = min(max_estimated_steps, self.trainer.max_steps) if self.trainer.max_steps != -1 else max_estimated_steps
        return max_estimated_steps