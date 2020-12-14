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
    datamodule.setup(stage='fit')
    dataloader = datamodule.val_dataloader()
    with torch.no_grad():
        for i, (batch, event_path) in tqdm(enumerate(zip(dataloader, dataloader.dataset.events))):
            inputs = batch['features'].to(model.device)
            labels = batch['labels'].to(model.device)
            prediction = model(inputs).cpu()

            event_name = event_path.stem
            output_path = experiment.run_prediction_dir / event_name
            inds, labels = dataloader.dataset.get_inds_labels(i)
            np.savez_compressed(
                output_path, prediction=prediction, inds=inds, inputs=inputs.F.cpu(), labels=labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('wandb_run_dir')
    args = parser.parse_args()
    run_path = Path(args.wandb_run_dir)
    experiment = Experiment(run_path)
    save_predictions(experiment)


if __name__ == '__main__':
    main()
