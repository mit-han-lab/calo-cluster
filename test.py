import argparse
import logging
from pathlib import Path
import pstats
import io

import hydra
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf

from calo_cluster.evaluation.experiments.base_experiment import BaseExperiment
from calo_cluster.evaluation.experiments.base_offset_experiment import BaseOffsetExperiment
from calo_cluster.utils.comm import is_rank_zero
import pytorch_lightning as pl
from torch.utils.data import Subset
import torch
from pytorch_lightning.profiler import SimpleProfiler

def test(cfg: DictConfig) -> None:
    experiment = hydra.utils.instantiate(cfg.experiment, )
    callbacks = []
    path = (experiment.plots_dir / cfg.study.path)
    path.mkdir(exist_ok=True, parents=True)
    logging.info(f'writing test data to {path}')
    study_callback = hydra.utils.instantiate(cfg.study, path=path)
    callbacks.append(study_callback)
    trainer = pl.Trainer(gpus=1, deterministic=True, callbacks=callbacks, precision=32)
    if cfg.n is not None:
        experiment.datamodule.test_dataset.files = experiment.datamodule.test_dataset.files[:cfg.n]

    if cfg.profile:
        import cProfile
        pr = cProfile.Profile()

        # profiled method
        pr = pr.runctx('trainer.test(experiment.model, datamodule=experiment.datamodule)', {'experiment': experiment, 'trainer': trainer}, {})
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
        ps.print_stats()

        with open('/global/homes/s/schuya/calo-cluster/profile.txt', 'w+') as f:
            f.write(s.getvalue())
    else:
        trainer.test(experiment.model, datamodule=experiment.datamodule)

@ hydra.main(config_path="test_configs", config_name="config")
def hydra_main(cfg: DictConfig) -> None:
    # Set up python logging.
    logger=logging.getLogger()
    if is_rank_zero():
        logger.setLevel(cfg.log_level)
        logging.info(OmegaConf.to_yaml(cfg))

    test(cfg)

if __name__ == '__main__':
    hydra_main()