from pathlib import Path
from typing import Union

import wandb
from omegaconf import OmegaConf

from ..models.spvcnn import SPVCNN


class Experiment():
    def __init__(self, outputs_dir: Path, predictions_dir: Path, entity: str = None, project: str = None, version: str = None):
        self.outputs_dir = outputs_dir
        self.predictions_dir = predictions_dir
        self.run_prediction_dir = self.predictions_dir / version
        self.run_prediction_dir.mkdir(exist_ok=True, parents=True)
        self.wandb_dir = self.outputs_dir / 'wandb'
        self.hydra_dir = self.outputs_dir / 'hydra'
        self.entity = entity
        self.project = project
        self.version = version

        self._model = None
        self._has_init = False

        self.load()

    def load(self):
        self.run_dir = [p for p in sorted(
            self.wandb_dir.glob(f'*{self.version}'))][-1]
        ckpt_dir = self.run_dir / 'files' / self.project / self.version / 'checkpoints'
        ckpt_path = [p for p in sorted(ckpt_dir.glob('*.ckpt'))][-1]
        self._model = SPVCNN.load_from_checkpoint(str(ckpt_path))


    @property
    def model(self):
        if self._model is not None:
            return self._model
        else:
            raise RuntimeError('model not loaded!')

    def _init(self):
        if not self._has_init:
            wandb.init(id=self.version, entity=self.entity,
                       project=self.project, resume='allow')
            self._has_init = True

    def save(self, filepath):
        self._init()
        wandb.save(filepath)
