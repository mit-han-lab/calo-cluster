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
import submitit
import torch
from omegaconf import DictConfig, OmegaConf

import wandb
from utils.comm import *


def add(a, b):
    return a + b

def train(cfg: DictConfig, output_dir: Path) -> None:
    logging.info('Beginning training...')
    datamodule = hydra.utils.instantiate(cfg.dataset)
    # Instantiate the model (pass configs to avoid pickle issues in checkpointing).
    semantic_criterion_cfg = None
    embed_criterion_cfg = None
    if 'semantic' in cfg.criterion:
        semantic_criterion_cfg = cfg.criterion.semantic
    if 'embed' in cfg.criterion:
        embed_criterion_cfg = cfg.criterion.embed
    model = hydra.utils.instantiate(cfg.model, optimizer_cfg=cfg.optimizer,
                                    scheduler_cfg=cfg.scheduler, semantic_criterion_cfg=semantic_criterion_cfg, embed_criterion_cfg=embed_criterion_cfg, metrics_cfg=cfg.metrics)

    # Set up checkpointing.
    if cfg.init_ckpt is not None:
        logging.info(f'Loading checkpoint={cfg.init_ckpt}')
        resume_from_checkpoint = cfg.init_ckpt
    else:
        resume_from_checkpoint = None
    checkpoint_callback = hydra.utils.instantiate(
        cfg.checkpoint, filepath=f'{str(output_dir)}/{{epoch:02d}}')

    # Set up wandb logging.
    if cfg.wandb.active:
        wandb_id = cfg.wandb.id
        if wandb_id is None:
            wandb_id = (output_dir.parent.name + output_dir.name).replace('-', '')
        logger = pl.loggers.WandbLogger(
            project=cfg.wandb.project, save_dir=str(output_dir), id=wandb_id, name=cfg.wandb.name, entity=cfg.wandb.entity)
    else:
        logger = True

    # train
    trainer = pl.Trainer(gpus=cfg.train.gpus, logger=logger, weights_save_path=str(
        output_dir), max_epochs=cfg.train.num_epochs, checkpoint_callback=checkpoint_callback, resume_from_checkpoint=resume_from_checkpoint, deterministic=True, distributed_backend=cfg.train.distributed_backend)
    if cfg.wandb.active:
        trainer.logger.log_hyperparams(cfg._content)
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
        job = executor.submit(train, cfg=cfg, output_dir=Path.cwd())
        logging.info(f'submitted job {job.job_id}.')
    else:
        train(cfg, output_dir=Path.cwd())


if __name__ == '__main__':
    hydra_main()  # pylint: disable=no-value-for-parameter
