from pathlib import Path
from typing import Union

import hydra
import numpy as np
import torch
import wandb
from hgcal_dev.visualization.hcal import HCalEvent
from hgcal_dev.visualization.hgcal import HGCalEvent
from hgcal_dev.visualization.vertex import VertexEvent
from omegaconf import OmegaConf
from tqdm import tqdm

from ..models.spvcnn import SPVCNN


class Experiment():
    def __init__(self, wandb_version, ckpt_name=None):
        run_path = self.get_run_path(wandb_version)
        self.run_path = run_path
        cfg_path = run_path / 'files' / '.hydra' / 'config.yaml'
        self.cfg = OmegaConf.load(cfg_path)

        self.run_prediction_dir = Path(
            self.cfg.predictions_dir) / self.cfg.wandb.version
        self.run_prediction_dir.mkdir(exist_ok=True, parents=True)

        ckpt_dir = self.cfg.outputs_dir / self.cfg.wandb.project / \
            self.cfg.wandb.version / 'checkpoints'
        if ckpt_name is None:
            ckpt_path = [p for p in sorted(ckpt_dir.glob('*.ckpt'))][-1]
            print(f'no checkpoint name given, using {ckpt_path}.')
        else:
            ckpt_path = ckpt_dir / ckpt_name
            if not ckpt_path.exists():
                raise RuntimeError(f'No checkpoint found at {ckpt_path}!')
        self.model = SPVCNN.load_from_checkpoint(str(ckpt_path))

        self.datamodule = hydra.utils.instantiate(self.cfg.dataset)
        self.datamodule.batch_size = 1
        self.datamodule.prepare_data()
        self.datamodule.setup(stage=None)

    def get_run_path(wandb_version):
        wandb_dir = Path('/home/alexj/outputs/wandb')
        for run_dir in wandb_dir.iterdir():
            if run_dir.stem.contains(wandb_version):
                return run_dir
        raise RuntimeError(
            f'run with wandb_version={wandb_version} not found; is {wandb_dir} correct?')

    def save_predictions(self):
        model = self.model
        datamodule = self.datamodule
        model.cuda(0)
        model.eval()
        datamodule.batch_size = 1
        datamodule.prepare_data()
        datamodule.setup(stage=None)
        for split, dataloader in zip(('test', 'val', 'train'), (datamodule.test_dataloader(), datamodule.val_dataloader(), datamodule.train_dataloader())):
            output_dir = self.run_prediction_dir / split
            output_dir.mkdir(exist_ok=True, parents=True)
            with torch.no_grad():
                for i, (batch, event_path) in tqdm(enumerate(zip(dataloader, dataloader.dataset.events))):
                    features = batch['features'].to(model.device)
                    inverse_map = batch['inverse_map'].F.type(torch.int)
                    event_name = event_path.stem
                    output_path = output_dir / event_name
                    if self.cfg.criterion.task == 'instance':
                        embedding = model(features).cpu().numpy()[inverse_map]
                        np.savez_compressed(output_path, embedding=embedding)
                    elif self.cfg.criterion.task == 'semantic':
                        labels = torch.argmax(model(features), dim=1).cpu().numpy()[
                            inverse_map]
                        np.savez_compressed(output_path, labels=labels)
                    elif self.cfg.criterion.task == 'panoptic':
                        out_c, out_e = model(features)
                        labels = torch.argmax(out_c, dim=1).cpu().numpy()[
                            inverse_map]
                        embedding = out_e.cpu().numpy()[inverse_map]
                        np.savez_compressed(
                            output_path, labels=labels, embedding=embedding)

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
            elif self.cfg.dataset._target_ == 'hgcal_dev.datasets.hgcal.HGCalDataModule':
                events.append(HGCalEvent(input_path, pred_path))
            elif self.cfg.dataset._target_ == 'hgcal_dev.datasets.vertex.VertexDataModule':
                events.append(VertexEvent(input_path, pred_path))
            else:
                raise NotImplementedError()
        return events
