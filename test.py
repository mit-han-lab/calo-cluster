import argparse
import copy
import logging
import os
import random
import shutil
import sys
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import wandb
from hgcal_dev.utils.experiment import Experiment


def save_predictions(experiment):
    model = experiment.model
    datamodule = experiment.datamodule
    model.cuda(0)
    model.eval()
    datamodule.batch_size = 1
    datamodule.prepare_data()
    datamodule.setup(stage=None)
    for split, dataloader in zip(('test', 'val', 'train'), (datamodule.test_dataloader(), datamodule.val_dataloader(), datamodule.train_dataloader())):
        output_dir = experiment.run_prediction_dir / split
        output_dir.mkdir(exist_ok=True, parents=True)
        with torch.no_grad():
            for i, (batch, event_path) in tqdm(enumerate(zip(dataloader, dataloader.dataset.events))):
                features = batch['features'].to(model.device)
                inverse_map = batch['inverse_map'].F.type(torch.int)
                event_name = event_path.stem
                output_path = output_dir / event_name
                if experiment.cfg.criterion.task == 'instance':
                    embedding = model(features).cpu().numpy()[inverse_map]
                    np.savez_compressed(output_path, embedding=embedding)
                elif experiment.cfg.criterion.task == 'class':
                    labels = torch.argmax(model(features), dim=1).cpu().numpy()[inverse_map]
                    np.savez_compressed(output_path, labels=labels)
                elif experiment.cfg.criterion.task == 'panoptic':
                    out_c, out_e = model(features)
                    labels = torch.argmax(out_c, dim=1).cpu().numpy()[inverse_map]
                    embedding = out_e.cpu().numpy()[inverse_map]
                    np.savez_compressed(output_path, labels=labels, embedding=embedding)
                


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('wandb_run_dir')
    parser.add_argument('--ckpt_path')
    args = parser.parse_args()
    run_path = Path(args.wandb_run_dir)
    if args.ckpt_path is not None:
        ckpt_path = Path(args.ckpt_path)
        experiment = Experiment(run_path, ckpt_path)
    else:
        experiment = Experiment(run_path)
    save_predictions(experiment)


if __name__ == '__main__':
    main()
