import time
from collections import OrderedDict
from typing import Callable, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.metrics import Metric

from ..modules.efficient_minkowski.functionals import *
from ..modules.efficient_minkowski.point_tensor import *
from ..modules.efficient_minkowski.sparse_tensor import *

import hydra

# z: PointTensor
# return: SparseTensor
def initial_voxelize(z, init_res, after_res):
    new_float_coord = torch.cat(
        [(z.C[:, :3] * init_res) / after_res, z.C[:, -1].view(-1, 1)], 1)

    pc_hash = hash_gpu(torch.floor(new_float_coord).int())
    sparse_hash = torch.unique(pc_hash)
    idx_query = sparse_hash_query(pc_hash, sparse_hash)
    counts = count_gpu(idx_query, len(sparse_hash))

    inserted_coords = voxelize_gpu(torch.floor(new_float_coord), idx_query,
                                   counts)
    inserted_coords = torch.round(inserted_coords).int()
    inserted_feat = voxelize_gpu(z.F, idx_query, counts)

    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)
    new_tensor.check()
    z.additional_features['idx_query'][1] = idx_query
    z.additional_features['counts'][1] = counts
    z.C = new_float_coord

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: SparseTensor
def point_to_voxel(x, z):
    if z.additional_features is None or z.additional_features.get('idx_query') is None\
       or z.additional_features['idx_query'].get(x.s) is None:
        #pc_hash = hash_gpu(torch.floor(z.C).int())
        pc_hash = hash_gpu(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s).int() * x.s,
                z.C[:, -1].int().view(-1, 1)
            ], 1))
        sparse_hash = hash_gpu(x.C)
        idx_query = sparse_hash_query(pc_hash, sparse_hash)
        counts = count_gpu(idx_query, x.C.shape[0])
        z.additional_features['idx_query'][x.s] = idx_query
        z.additional_features['counts'][x.s] = counts
    else:
        idx_query = z.additional_features['idx_query'][x.s]
        counts = z.additional_features['counts'][x.s]

    inserted_feat = voxelize_gpu(z.F, idx_query, counts)
    new_tensor = SparseTensor(inserted_feat, x.C, x.s)
    new_tensor.coord_maps = x.coord_maps
    new_tensor.kernel_maps = x.kernel_maps

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: PointTensor
def voxel_to_point(x, z, nearest=False):
    if z.idx_query is None or z.weights is None or z.idx_query.get(
            x.s) is None or z.weights.get(x.s) is None:
        kr = KernelRegion(2, x.s, 1)
        off = kr.get_kernel_offset().to(z.F.device)
        #old_hash = kernel_hash_gpu(torch.floor(z.C).int(), off)
        old_hash = kernel_hash_gpu(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s).int() * x.s,
                z.C[:, -1].int().view(-1, 1)
            ], 1), off)
        pc_hash = hash_gpu(x.C.to(z.F.device))
        idx_query = sparse_hash_query(old_hash, pc_hash)
        weights = calc_ti_weights(z.C, idx_query,
                                  scale=x.s).transpose(0, 1).contiguous()
        idx_query = idx_query.transpose(0, 1).contiguous()
        if nearest:
            weights[:, 1:] = 0.
            idx_query[:, 1:] = -1
        new_feat = devoxelize(x.F, idx_query, weights)
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features
        new_tensor.idx_query[x.s] = idx_query
        new_tensor.weights[x.s] = weights
        z.idx_query[x.s] = idx_query
        z.weights[x.s] = weights

    else:
        new_feat = devoxelize(x.F, z.idx_query.get(x.s), z.weights.get(x.s))
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features

    return new_tensor


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            MinkowskiConvolution(inc,
                                 outc,
                                 kernel_size=ks,
                                 dilation=dilation,
                                 stride=stride), MinkowskiBatchNorm(outc),
            MinkowskiReLU(True))

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            MinkowskiConvolution(inc,
                                 outc,
                                 kernel_size=ks,
                                 stride=stride,
                                 transpose=True), MinkowskiBatchNorm(outc),
            MinkowskiReLU(True))

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            MinkowskiConvolution(inc,
                                 outc,
                                 kernel_size=ks,
                                 dilation=dilation,
                                 stride=stride), MinkowskiBatchNorm(outc),
            MinkowskiReLU(True),
            MinkowskiConvolution(outc,
                                 outc,
                                 kernel_size=ks,
                                 dilation=dilation,
                                 stride=1), MinkowskiBatchNorm(outc))

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                MinkowskiConvolution(
                    inc, outc, kernel_size=1, dilation=1, stride=stride),
                MinkowskiBatchNorm(outc)
        )

        self.relu = MinkowskiReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class SPVCNN(pl.LightningModule):
    def __init__(self, cr: float, cs: List[int], pres: float, vres: float, num_classes: int, embed_dim: int, metrics_cfg: dict, optimizer_cfg: dict, scheduler_cfg: dict, embed_criterion_cfg: dict, semantic_criterion_cfg: dict, head: str):
        super().__init__()
        assert head in ('instance', 'class', 'class_and_instance')
        cs = [int(cr * x) for x in cs]
        self.pres = pres
        self.vres = vres
        self.head = head
        self.save_hyperparameters()
        self.metrics = hydra.utils.call(metrics_cfg)
        self.optimizer_factory = hydra.utils.instantiate(optimizer_cfg)
        self.scheduler_factory = hydra.utils.instantiate(scheduler_cfg)
        if head == 'instance' or head == 'class_and_instance':
            self.embed_criterion = hydra.utils.instantiate(embed_criterion_cfg)
        if head == 'class' or head == 'class_and_instance':
            self.semantic_criterion = hydra.utils.instantiate(semantic_criterion_cfg)

        self.stem = nn.Sequential(
            MinkowskiConvolution(5, cs[0], kernel_size=3, stride=1),
            MinkowskiBatchNorm(cs[0]), MinkowskiReLU(True),
            MinkowskiConvolution(cs[0], cs[0], kernel_size=3, stride=1),
            MinkowskiBatchNorm(cs[0]), MinkowskiReLU(True))

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

        if head == 'class' or head == 'class_and_instance':
            self.classifier = nn.Sequential(nn.Linear(cs[8],
                                                    num_classes))
        if head == 'instance' or head == 'class_and_instance':
            self.embedder = nn.Sequential(nn.Linear(cs[8],
                                                embed_dim))

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
        self.cat = MinkowskiConcatenation()

        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: SparseTensor z: PointTensor
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)
        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F  # + self.point_transforms[0](z.F)

        x1 = point_to_voxel(x0, z0)
        x1 = self.stage1(x1)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        # point transform 32 to 256
        z1 = voxel_to_point(x4, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)

        y1 = point_to_voxel(x4, z1)
        y1.F = self.dropout(y1.F)
        y1 = self.up1[0](y1)
        y1 = self.cat([y1, x3])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = self.cat([y2, x2])
        y2 = self.up2[1](y2)
        # point transform 256 to 128
        z2 = voxel_to_point(y2, z1)
        z2.F = z2.F + self.point_transforms[1](z1.F)

        y3 = point_to_voxel(y2, z2)
        y3.F = self.dropout(y3.F)
        y3 = self.up3[0](y3)
        y3 = self.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = self.cat([y4, x0])
        y4 = self.up4[1](y4)
        z3 = voxel_to_point(y4, z2)
        z3.F = z3.F + self.point_transforms[2](z2.F)

        if self.head == 'class':
            out = self.classifier(z3.F)
        elif self.head == 'instance':
            out = self.embedder(z3.F)
        elif self.head == 'class_and_instance':
            out = (self.classifier(z3.F), self.embedder(z3.F))
        
        return out

    def configure_optimizers(self):
        optimizer = self.optimizer_factory(self.parameters())
        if self.scheduler_factory is not None:
            scheduler = self.scheduler_factory(optimizer)
            return [optimizer], [scheduler]
        else:
            return optimizer

    def step(self, batch, batch_idx, split):
        (locs, feats, targets), all_labels, invs = batch
        inputs = SparseTensor(feats, coords=locs)
        targets = targets.long()
        outputs = self(inputs)
        if isinstance(outputs, SparseTensor):
            outputs = outputs.F
        if self.head == 'class':
            loss = self.semantic_criterion(outputs, targets)
        elif self.head == 'instance':
            loss = self.embed_criterion(outputs, targets)
        elif self.head == 'class_and_instance':
            class_loss = self.semantic_criterion(outputs[0], targets[:, 0])
            embed_loss = 10 * self.embed_criterion(outputs[1], targets[:, 1])
            loss = class_loss + embed_loss
        if split == 'train':
            result = pl.TrainResult(loss)
        else:
            result = pl.EvalResult(checkpoint_on=loss)
        result.log(f'{split}_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        if self.head == 'class_and_instance':
            result.log(f'{split}_class_loss', class_loss, sync_dist=True, on_epoch=True, on_step=False)
            result.log(f'{split}_embed_loss', embed_loss, sync_dist=True, on_epoch=True, on_step=False)
        # Hack to record 
        if split == 'test':
            if batch_idx == 0:
                if self.head == 'class_and_instance':
                    self.class_predictions = []
                    self.embed_predictions = []
                else:
                    self.predictions = []
                self.locs = []
                self.feats = []
                self.targets = []
            if self.head == 'class_and_instance':
                self.class_predictions.append(outputs[0].cpu().numpy())
                self.embed_predictions.append(outputs[1].cpu().numpy())
            else:
                self.predictions.append(outputs.cpu().numpy())
            self.locs.append(locs.cpu().numpy())
            self.feats.append(feats.cpu().numpy())
            self.targets.append(targets.cpu().numpy())
        return result

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, split='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, split='val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, split='test')