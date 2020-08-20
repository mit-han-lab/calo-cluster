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
from omegaconf import DictConfig, OmegaConf
from test_tube.hpc import SlurmCluster

import wandb
from utils.comm import *


def train(cfg: DictConfig) -> None:
    datamodule = hydra.utils.instantiate(cfg.dataset)
    # Instantiate the model (pass configs to avoid pickle issues in checkpointing).
    model = hydra.utils.instantiate(cfg.model, optimizer_cfg=cfg.optimizer,
                                    scheduler_cfg=cfg.scheduler, criterion_cfg=cfg.criterion, metrics_cfg=cfg.metrics)

    # Set up checkpointing.
    if cfg.init_ckpt is not None:
        logging.info(f'Loading checkpoint={cfg.init_ckpt}')
        resume_from_checkpoint = cfg.init_ckpt
    else:
        resume_from_checkpoint = None
    checkpoint_callback = hydra.utils.instantiate(
        cfg.checkpoint, filepath=f'{os.getcwd()}/{{epoch:02d}}')

    # Set up wandb logging.
    if cfg.wandb.active:
        logger = pl.loggers.WandbLogger(
            project=cfg.wandb.project, save_dir=os.getcwd())
    else:
        logger = True

    # train
    trainer = pl.Trainer(gpus=cfg.train.gpus, logger=logger, weights_save_path=os.getcwd(
    ), max_epochs=cfg.train.num_epochs, checkpoint_callback=checkpoint_callback, resume_from_checkpoint=resume_from_checkpoint)
    if cfg.wandb.active:
        trainer.logger.experiment.config.update(cfg._content)
    trainer.fit(model=model, datamodule=datamodule)


@hydra.main(config_path="configs", config_name="config")
def hydra_main(cfg: DictConfig) -> None:
    # Set up python logging. 
    logger = logging.getLogger()
    logger.setLevel(cfg.log_level)
    logging.info(cfg.pretty())
    # Workaround to fix hydra + pytorch-lightning (see https://github.com/PyTorchLightning/pytorch-lightning/issues/2727)
    if cfg.train.distributed_backend == 'ddp':
        cwd = os.getcwd()
        sys.argv = sys.argv[:1]
        sys.argv.extend([
            f"hydra.run.dir={cwd}",
            "hydra/hydra_logging=disabled",
            "hydra/job_logging=disabled",
        ])
        overrides = OmegaConf.load('.hydra/overrides.yaml')
        sys.argv.extend(
            (o for o in overrides if not 'hydra/sweeper' in o and not 'hydra/launcher' in o))
    train(cfg)


if __name__ == '__main__':
    hydra_main()  # pylint: disable=no-value-for-parameter
