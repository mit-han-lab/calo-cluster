from pathlib import Path
from typing import Union

import wandb
from omegaconf import OmegaConf

from ..models.spvcnn import SPVCNN
import hydra

class Experiment():
    def __init__(self, run_path, ckpt_path=None):
        cfg_path = run_path / 'files' / '.hydra' / 'config.yaml'
        self.cfg = OmegaConf.load(cfg_path)

        self.run_prediction_dir = Path(self.cfg.predictions_dir) / self.cfg.wandb.version
        self.run_prediction_dir.mkdir(exist_ok=True, parents=True)

        ckpt_dir = run_path / 'files' / self.cfg.wandb.project / \
            self.cfg.wandb.version / 'checkpoints'
        if ckpt_path is None:
            ckpt_path = [p for p in sorted(ckpt_dir.glob('*.ckpt'))][-1]
        self.model = SPVCNN.load_from_checkpoint(str(ckpt_path))

        self.datamodule = hydra.utils.instantiate(self.cfg.dataset)
        self.datamodule.batch_size = 1
        self.datamodule.prepare_data()
        self.datamodule.setup(stage=None)


        wandb.init(id=self.cfg.wandb.version, entity=self.cfg.wandb.entity,
                   project=self.cfg.wandb.project, resume='allow')

    def save(self, filepath):
        wandb.save(filepath)
