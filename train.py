import argparse
import copy
import os
import random
import shutil
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

import wandb
from utils.comm import *


@hydra.main(config_path="configs", config_name="config")
def hydra_main(cfg: DictConfig) -> None:
    print(cfg.pretty())
    datamodule = hydra.utils.instantiate(cfg.dataset)
    metrics = hydra.utils.call(cfg.metrics)
    optimizer_factory = hydra.utils.instantiate(cfg.optimizer)
    scheduler_factory = hydra.utils.instantiate(cfg.scheduler)
    criterion = hydra.utils.instantiate(cfg.criterion)
    model = hydra.utils.instantiate(cfg.model, optimizer_factory=optimizer_factory,
                                    scheduler_factory=scheduler_factory, criterion=criterion, metrics=metrics)
    checkpoint_callback = hydra.utils.instantiate(
        cfg.train.checkpoint, filepath=os.getcwd())
    wandb_logger = pl.loggers.WandbLogger(
        project="hgcal-spvcnn", save_dir=os.getcwd())
    wandb.config = cfg._content
    trainer = pl.Trainer(gpus=1, logger=wandb_logger, weights_save_path=os.getcwd(
    ), max_epochs=cfg.train.num_epochs, checkpoint_callback=checkpoint_callback)
    trainer.fit(model=model, datamodule=datamodule)

if __name__ == '__main__':
    hydra_main()  # pylint: disable=no-value-for-parameter
