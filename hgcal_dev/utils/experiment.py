from pathlib import Path
from typing import Union

import hydra
import wandb
from hgcal_dev.visualization.hcal import HCalEvent
from hgcal_dev.visualization.hgcal import HGCalEvent
from omegaconf import OmegaConf
from tqdm import tqdm

from ..models.spvcnn import SPVCNN


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

    def get_events(self, split, n=-1):
        if split == 'train':
            dataloader = self.datamodule.train_dataloader()
        elif split == 'val':
            dataloader = self.datamodule.val_dataloader()
        elif split == 'test':
            dataloader = self.datamodule.test_dataloader()
        else:
            raise NotImplementedError()
        if n == -1:
            input_paths = dataloader.dataset.events
        else:
            input_paths = dataloader.dataset.events[:n]
        events = []
        for input_path in tqdm(input_paths):
            pred_dir = self.run_prediction_dir / split
            event_name = input_path.stem
            pred_path = pred_dir / f'{event_name}.npz'
            if self.cfg.dataset._target_ == 'hgcal_dev.datasets.hcal.HCalDataModule':
                events.append(HCalEvent(input_path, pred_path))
            else:
                events.append(HGCalEvent(input_path, pred_path))
        return events

    def save(self, filepath):
        wandb.save(filepath)
