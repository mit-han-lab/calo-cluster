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
from omegaconf import OmegaConf, DictConfig

import wandb
from hgcal_dev.modules.efficient_minkowski.sparse_tensor import *
from hgcal_dev.utils.experiment import Experiment
from tqdm import tqdm


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


def save_predictions(experiment, datamodule):
    model = experiment.model
    model.cuda(0)
    model.eval()

    datamodule.batch_size = 1
    datamodule.prepare_data()
    datamodule.setup(stage='fit')
    dataloader = datamodule.val_dataloader()
    with torch.no_grad():
        for i, (batch, event_path) in tqdm(enumerate(zip(dataloader, dataloader.dataset.events))):
            if i == 5:
                breakpoint()
            (locs, feats, targets), all_labels, invs = batch
            inputs = SparseTensor(feats, coords=locs).to(model.device)
            prediction = model(inputs).cpu()

            event_name = event_path.stem
            output_path = experiment.run_prediction_dir / event_name
            inds, labels = dataloader.dataset.get_inds_labels(i)
            np.savez_compressed(
                output_path, prediction=prediction, inds=inds, labels=labels)


@hydra.main(config_path="configs", config_name="config")
def hydra_main(cfg: DictConfig) -> None:
    experiment = Experiment(Path(cfg.outputs_dir), Path(cfg.predictions_dir), entity=cfg.wandb.entity, project=cfg.wandb.project, version=cfg.wandb.version)
    datamodule = hydra.utils.instantiate(cfg.dataset)
    save_predictions(experiment, datamodule)
    #trainer = pl.Trainer()

if __name__ == '__main__':
    hydra_main()  # pylint: disable=no-value-for-parameter
