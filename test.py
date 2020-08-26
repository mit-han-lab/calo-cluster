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
from modules.efficient_minkowski.sparse_tensor import *


def load_from_run(run_dir: str):
    run_dir = Path(run_dir)
    config_path = run_dir / '.hydra' / 'config.yaml'
    config = OmegaConf.load(config_path)
    checkpoint = str([p for p in sorted(run_dir.glob('*.ckpt'))][-1])
    model = config['model']['target']
    if model == 'models.spvcnn.SPVCNN':
        from models.spvcnn import SPVCNN
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
    batch_size = datamodule.batch_size
    datamodule.setup(stage='test')
    test_loader = datamodule.test_dataloader()
    dataset = test_loader.dataset
    trainer = pl.Trainer(gpus=1)
    trainer.test(model=model, datamodule=datamodule)
    predictions = np.concatenate(trainer.model.predictions)
    index = 0
    for i, event_path in enumerate(dataset.events):
        n = dataset[i][0].shape[0]
        event_name = event_path.stem
        output_path = output_dir / event_name
        prediction = predictions[index:index+n, :]
        inds = dataset.get_inds(i)
        labels = dataset[i][2]
        np.savez_compressed(output_path, prediction=prediction, inds=inds, labels=labels)
        index += n
    assert index == predictions.shape[0]


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
