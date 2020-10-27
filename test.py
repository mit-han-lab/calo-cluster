import argparse
import copy
import logging
import os
import random
import shutil
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import yaml
from omegaconf import OmegaConf

import wandb
from hgcal_dev.modules.efficient_minkowski.sparse_tensor import *


def load_from_run(run_dir: str):
    run_dir = Path(run_dir)
    config_path = run_dir / '.hydra' / 'config.yaml'
    config = OmegaConf.load(config_path)
    checkpoint = str([p for p in sorted(run_dir.glob('*.ckpt'))][-1])
    model = config.model.target._target_
    if model == 'hgcal_dev.models.spvcnn.SPVCNN':
        from hgcal_dev.models.spvcnn import SPVCNN
        model = SPVCNN
    else:
        raise NotImplementedError()
    model = model.load_from_checkpoint(checkpoint)
    wandb_dir = run_dir / 'wandb'
    run_path = [p for p in sorted(wandb_dir.glob('run*'))][-1]
    wandb_id = run_path.name.split('-')[-1]
    project = config['wandb']['project']
    datamodule = hydra.utils.instantiate(config.dataset)
    return model, checkpoint, project, wandb_id, datamodule


def save_predictions(model, datamodule, output_dir: Path):
    torch.cuda.set_device(0)
    model.cuda(0)
    model.zero_grad()
    torch.set_grad_enabled(False)

    datamodule.setup(stage='test')
    test_loader = datamodule.test_dataloader()
    dataset = test_loader.dataset
    
    for i, (data, event_path) in enumerate(zip(dataset, dataset.events)):
        locs, feats, _, _, _ = data
        feats = torch.as_tensor(feats).to(model.device)
        locs = torch.as_tensor(locs).to(model.device)
        inputs = SparseTensor(feats, coords=locs)
        prediction = model(inputs).cpu()

        event_name = event_path.stem
        output_path = output_dir / event_name
        inds, labels = dataset.get_inds_labels(i)
        np.savez_compressed(output_path, prediction=prediction, inds=inds, labels=labels)


def main() -> None:
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('run_dir')
    parser.add_argument('output_dir')
    args = parser.parse_args()
    model, checkpoint, project, wandb_id, datamodule = load_from_run(
        args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    save_predictions(model, datamodule, output_dir)


if __name__ == '__main__':
    main()
