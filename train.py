from calo_cluster.utils.comm import is_rank_zero
import logging
import shutil
from pathlib import Path

import hydra
import pytorch_lightning as pl
import submitit
import yaml
from omegaconf import DictConfig, OmegaConf, open_dict
from calo_cluster.models.spvcnn import SPVCNN

from calo_cluster.training.config import fix_config

def train(cfg: DictConfig) -> None:
    logging.info('Beginning training...')

    fix_config(cfg)

    if cfg.overfit:
        overfit_batches = 1
        cfg.train.batch_size = 1
        cfg.checkpoint.save_top_k = 0
        cfg.checkpoint.save_last = False
    else:
        overfit_batches = 0.0
    
    callbacks = []


    # Set up SWA.
    if cfg.swa.active:
        swa_callback = hydra.utils.instantiate(cfg.swa.callback)
        callbacks.append(swa_callback)

    # Set up checkpointing.
    if cfg.resume_ckpt is not None:
        logging.info(f'Resuming checkpoint={cfg.resume_ckpt}')
        resume_from_checkpoint = cfg.resume_ckpt
    else:
        resume_from_checkpoint = None
    checkpoint_callback = hydra.utils.instantiate(cfg.checkpoint)
    callbacks.append(checkpoint_callback)

    # Set up learning rate monitor.
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # Set up wandb logging.
    logger = hydra.utils.instantiate(
        cfg.wandb, save_dir=cfg.outputs_dir, version=cfg.wandb.version, group=cfg.wandb.name)
    if is_rank_zero():
        shutil.copytree(Path.cwd() / '.hydra',
                        Path(logger.experiment.dir) / '.hydra')
    cfg.wandb.version = logger.version

    if is_rank_zero():
        config_path = Path(logger.experiment.dir) / '.hydra' / 'config.yaml'
        with config_path.open('r+') as f:
            data = yaml.load(f, Loader=yaml.CLoader)
            data['wandb']['version'] = cfg.wandb.version
            f.seek(0)
            yaml.dump(data, f)

    datamodule = hydra.utils.instantiate(cfg.dataset)
    if cfg.init_ckpt is not None:
        model = SPVCNN.load_from_checkpoint(cfg.init_ckpt, **cfg)
    else:
        model = hydra.utils.instantiate(cfg.model.target, cfg)
    
    # train
    trainer = pl.Trainer(gpus=cfg.train.gpus, logger=logger, max_epochs=cfg.train.num_epochs, resume_from_checkpoint=resume_from_checkpoint, deterministic=True, accelerator=cfg.train.distributed_backend, overfit_batches=overfit_batches, val_check_interval=cfg.val_check_interval, callbacks=callbacks, precision=32, log_every_n_steps=1)
    if is_rank_zero():
        trainer.logger.log_hyperparams(cfg._content)  # pylint: disable=no-member
    trainer.fit(model=model, datamodule=datamodule)


@hydra.main(config_path="configs", config_name="config")
def hydra_main(cfg: DictConfig) -> None:
    # Set up python logging.
    logger = logging.getLogger()
    if is_rank_zero():
        logger.setLevel(cfg.log_level)
        logging.info(OmegaConf.to_yaml(cfg))
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
