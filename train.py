import argparse
import copy
import logging
import os
import random
import shutil
import sys
from pathlib import Path

import torch

import hydra
import pytorch_lightning as pl
import submitit
import wandb
from omegaconf import DictConfig, OmegaConf


def add(a, b):
    return a + b


def train(cfg: DictConfig) -> None:
    logging.info('Beginning training...')

    if cfg.overfit:
        overfit_batches = 1
        cfg.train.batch_size = 1
    else:
        overfit_batches = 0.0

    # Set up checkpointing.
    if cfg.init_ckpt is not None:
        logging.info(f'Loading checkpoint={cfg.init_ckpt}')
        resume_from_checkpoint = cfg.init_ckpt
    else:
        resume_from_checkpoint = None
    checkpoint_callback = hydra.utils.instantiate(cfg.checkpoint)

    # Set up wandb logging.
    logger = hydra.utils.instantiate(
        cfg.wandb, save_dir=cfg.outputs_dir, version=cfg.wandb.version, group=cfg.wandb.name)
    shutil.copytree(Path.cwd() / '.hydra',
                    Path(logger.experiment.dir) / '.hydra')
    cfg.wandb.version = logger.version

    datamodule = hydra.utils.instantiate(cfg.dataset)
    model = hydra.utils.instantiate(cfg.model.target, cfg=cfg)

    # train
    trainer = pl.Trainer(gpus=cfg.train.gpus, logger=logger, max_epochs=cfg.train.num_epochs, checkpoint_callback=checkpoint_callback,
                         resume_from_checkpoint=resume_from_checkpoint, deterministic=True, distributed_backend=cfg.train.distributed_backend, overfit_batches=overfit_batches)
    trainer.logger.log_hyperparams(cfg._content)  # pylint: disable=no-member
    trainer.fit(model=model, datamodule=datamodule)


@hydra.main(config_path="configs", config_name="config")
def hydra_main(cfg: DictConfig) -> None:
    # Set up python logging.
    logger = logging.getLogger()
    logger.setLevel(cfg.log_level)
    logging.info(cfg.pretty())
    if 'slurm' in cfg.train:
        slurm_dir = Path.cwd() / 'slurm'
        slurm_dir.mkdir()
        executor = submitit.AutoExecutor(slurm_dir)
        executor.update_parameters(slurm_gpus_per_node=cfg.train.slurm.gpus_per_node, slurm_nodes=cfg.train.slurm.nodes, slurm_ntasks_per_node=cfg.train.slurm.gpus_per_node,
                                   slurm_cpus_per_task=cfg.train.slurm.cpus_per_task, slurm_time=cfg.train.slurm.time, slurm_additional_parameters={'constraint': 'gpu', 'account': cfg.train.slurm.account})
        job = executor.submit(train, cfg=cfg)
        logging.info(f'submitted job {job.job_id}.')
    else:
        train(cfg)


if __name__ == '__main__':
    hydra_main()  # pylint: disable=no-value-for-parameter
