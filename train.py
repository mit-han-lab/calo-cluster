import logging
import os
import shutil
from pathlib import Path

import hydra
import pytorch_lightning as pl
import submitit
import yaml
from omegaconf import DictConfig, OmegaConf

from calo_cluster.training.config import add_wandb_version, fix_task
from calo_cluster.utils.comm import is_rank_zero
import wandb
import io
import logging
import pstats
import warnings
from pytorch_lightning.callbacks import RichProgressBar

def train(cfg: DictConfig, code_dir: str) -> None:
    logging.info('Beginning training...')

    fix_task(cfg)

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

    # add rich progress bar
    if cfg.cluster.name == 'local':
        callbacks.append(RichProgressBar())

    # Set up learning rate monitor.
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # Set up wandb logging.
    logger = hydra.utils.instantiate(
        cfg.wandb, save_dir=cfg.outputs_dir, version=cfg.wandb.version, group=cfg.wandb.name)
    if is_rank_zero():
        # wandb needs the cwd to be within the git repository to store the current git hash.
        old_cwd = Path.cwd()
        os.chdir(code_dir)
        experiment_dir = Path(logger.experiment.dir) 
        os.chdir(old_cwd)

        # copy the .hydra files to the experiment dir.
        shutil.copytree(Path.cwd() / '.hydra',
                        experiment_dir / '.hydra')
    cfg.wandb.version = logger.version

    if is_rank_zero():
        config_path = Path(logger.experiment.dir) / '.hydra' / 'config.yaml'
        with config_path.open('r+') as f:
            data = yaml.load(f, Loader=yaml.CLoader)
            data['wandb']['version'] = cfg.wandb.version
            f.seek(0)
            yaml.dump(data, f)

    datamodule = hydra.utils.instantiate(cfg.dataset)
    model = hydra.utils.instantiate(cfg.model.target, cfg)

    #count_parameters(model)
    # train
    trainer = pl.Trainer(devices=cfg.train.devices, logger=logger, max_epochs=cfg.train.num_epochs, resume_from_checkpoint=resume_from_checkpoint, deterministic=cfg.deterministic, accelerator=cfg.train.accelerator, overfit_batches=overfit_batches, val_check_interval=cfg.val_check_interval, callbacks=callbacks, precision=32, log_every_n_steps=1, num_sanity_val_steps=0)
    if is_rank_zero():
        warnings.filterwarnings("ignore", category=UserWarning)
        trainer.logger.log_hyperparams(cfg._content)  # pylint: disable=no-member
    if cfg.profile:
        import cProfile
        pr = cProfile.Profile()

        # profiled method
        pr = pr.runctx('trainer.fit(model=model, datamodule=datamodule)', {'trainer': trainer, 'model': model, 'datamodule': datamodule}, {})
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
        ps.print_stats()

        with open('/global/homes/s/schuya/calo-cluster/profile.txt', 'w+') as f:
            f.write(s.getvalue())
    else:
        trainer.fit(model=model, datamodule=datamodule)

from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

@hydra.main(config_path="train_configs", config_name="config")
def hydra_main(cfg: DictConfig) -> None:
    # Set up python logging.
    logger = logging.getLogger()
    code_dir = hydra.utils.get_original_cwd()
    if is_rank_zero():
        logger.setLevel(cfg.log_level)
        wandb_version = wandb.util.generate_id()
        add_wandb_version(cfg, wandb_version)
    if cfg.cluster.name == 'slurm':
        slurm_dir = Path.cwd() / 'slurm'
        slurm_dir.mkdir()
        logging.info(f'Slurm logs: {slurm_dir}')
        executor = submitit.AutoExecutor(slurm_dir)
        executor.update_parameters(slurm_gpus_per_node=cfg.cluster.gpus_per_node, slurm_nodes=cfg.cluster.nodes, slurm_ntasks_per_node=cfg.cluster.gpus_per_node,
                                   slurm_cpus_per_task=cfg.cluster.cpus_per_task, slurm_time=cfg.cluster.time, slurm_additional_parameters={'constraint': 'gpu', 'account': cfg.cluster.account, 'requeue': True})
        job = executor.submit(train, cfg=cfg, code_dir=code_dir)
        logging.info(f'submitted job {job.job_id}.')
    else:
        train(cfg, code_dir)


if __name__ == '__main__':
    hydra_main()  # pylint: disable=no-value-for-parameter
