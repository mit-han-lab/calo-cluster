
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
from torchsparse.tensor import PointTensor, SparseTensor

from calo_cluster.utils.comm import is_rank_zero
import time

from .utils import *
import math

def conv3x3(in_planes, out_planes, stride=1):  # no padding now
    return spnn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, bias=True)


def conv1x3(in_planes, out_planes, stride=1):  # no padding now
    return spnn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride, bias=True)


def conv1x1x3(in_planes, out_planes, stride=1):  # no padding now
    return spnn.Conv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride, bias=True)


def conv1x3x1(in_planes, out_planes, stride=1):  # no padding now
    return spnn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride, bias=True)


def conv3x1x1(in_planes, out_planes, stride=1):  # no padding now
    return spnn.Conv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride, bias=True)


def conv3x1(in_planes, out_planes, stride=1):  # no padding now
    return spnn.Conv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride, bias=True)


def conv1x1(in_planes, out_planes, stride=1):  # no padding now
    return spnn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


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

        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[8] + cs[0], cs[8], ks=3, stride=1,
                              dilation=1),
                ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1),
            )
        ])

        if task == 'semantic' or task == 'panoptic':
            self.logits = conv3x3(cs[8], self.hparams.dataset.num_classes)

        # instance
        if task == 'instance' or task == 'panoptic':
            self.conv1 = conv3x3(cs[8], cs[8])
            self.bn1 = spnn.BatchNorm(cs[8])
            self.act1 = spnn.LeakyReLU()
            self.conv2 = conv3x3(cs[8], 2 * cs[8])
            self.bn2 = spnn.BatchNorm(2 * cs[8])
            self.act2 = spnn.LeakyReLU()
            self.conv3 = conv3x3(2 * cs[8], cs[8])
            self.bn3 = spnn.BatchNorm(cs[8])
            self.act3 = spnn.LeakyReLU()

            embed_dim = self.hparams.model.embed_dim
            self.offset = nn.Sequential(
                nn.Linear(cs[8] + embed_dim, cs[8], bias=True),
                nn.BatchNorm1d(cs[8]),
                nn.ReLU()
            )
            self.offset_linear = nn.Linear(cs[8], embed_dim, bias=True)

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
            ),
            nn.Sequential(
                nn.Linear(cs[6], cs[8]),
                nn.BatchNorm1d(cs[8]),
                nn.ReLU(True),
            )
        ])

        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def classifier(self, x):
        logits = self.logits(x)
        return logits.F

    def instance(self, x: SparseTensor, coordinates: torch.tensor):
        x = self.conv1(x)
        x = self.act1(self.bn1(x))
        x = self.conv2(x)
        x = self.act2(self.bn2(x))
        x = self.conv3(x)
        x = self.act3(self.bn3(x))

        x = self.offset_linear(self.offset(torch.cat([x.F, coordinates], dim=1)))
        return x

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def num_inf_or_nan(self, x):
        return (torch.isinf(x.F).sum(), torch.isnan(x.F).sum())

    def forward(self, inputs):
        x = inputs['features']
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

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)
        z3 = voxel_to_point(y4, z2)
        z3.F = z3.F + self.point_transforms[2](z2.F)
        z3 = SparseTensor(z3.F, x.C)

        task = self.hparams.task
        out = {}
        if task == 'semantic' or task == 'panoptic':
            out['pred_semantic_scores'] = self.classifier(z3)
            out['pred_semantic_labels'] = out['pred_semantic_scores'].argmax(dim=1)
        if task == 'instance' or task == 'panoptic':
            out['pred_offsets'] = self.instance(z3, inputs['coordinates'].F)
        return out