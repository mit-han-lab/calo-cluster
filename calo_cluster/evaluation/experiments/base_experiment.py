import logging
import math
from pathlib import Path
from re import L
from typing import Union

import cycler
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import torch
import wandb
import yaml
from calo_cluster.clustering.meanshift import MeanShift
from calo_cluster.models.spvcnn import SPVCNN
from omegaconf import OmegaConf
from plotly.subplots import make_subplots
from sklearn.metrics import auc, confusion_matrix, roc_curve
from tqdm.auto import tqdm
import importlib

from calo_cluster.training.config import fix_config


class BaseEvent():
    """Aggregates the input/output/clustering for a single event."""

    def __init__(self, input_path, pred_path: Path, task: str):
        self.input_path = input_path
        self.pred_path = pred_path
        self.task = task
        self._pred_instance_labels = None
        self._load()

    def _load(self):
        self.input_event = self._load_input()
        self._load_predictions()

    def _load_input(self):
        input_event = pd.read_pickle(self.input_path)
        return input_event

    def _load_predictions(self):
        event_prediction = np.load(self.pred_path)

        if self.task == 'panoptic':
            self.pred_semantic_labels = event_prediction['labels']
            self.embedding = event_prediction['embedding']
        elif self.task == 'instance':
            self.embedding = event_prediction['embedding']
        elif self.task == 'semantic':
            self.pred_semantic_labels = event_prediction['labels']


class BaseExperiment():
    def __init__(self, wandb_version, ckpt_name=None):
        self.wandb_version = wandb_version
            
        self.run_path = self.get_run_path(self.wandb_version)
        cfg_path = self.run_path / 'files' / '.hydra' / 'config.yaml'
        self.cfg = OmegaConf.load(cfg_path)
        fix_config(self.cfg)
        self.ckpt_path, self.ckpt_name = self._get_ckpt(
            self.cfg, ckpt_name)
        self.run_prediction_dir = Path(
            self.cfg.predictions_dir) / self.cfg.wandb.version / self.ckpt_name
        self.run_prediction_dir.mkdir(exist_ok=True, parents=True)
        self.model = self.load_model(str(self.ckpt_path))

        plots_dir = Path(self.cfg.plots_dir)
        self.plots_dir = plots_dir / self.wandb_version / self.ckpt_name

        self.datamodule = hydra.utils.instantiate(self.cfg.dataset)
        self.datamodule.prepare_data()
        self.datamodule.setup(stage=None)

        self.num_classes = self.datamodule.num_classes

        self.plots_dir.mkdir(exist_ok=True, parents=True)

        self.task = self.cfg.criterion.task

    def _get_ckpt(self, cfg, ckpt_name):
        ckpt_dir = Path(cfg.outputs_dir) / cfg.wandb.project / \
            cfg.wandb.version / 'checkpoints'
        if ckpt_name is None:
            ckpt_path = ckpt_dir / 'last.ckpt'
            if not ckpt_path.exists():
                ckpt_path = [p for p in sorted(ckpt_dir.glob('*.ckpt'))][-1]
            print(f'no checkpoint name given, using {ckpt_path}.')
        else:
            ckpt_path = ckpt_dir / ckpt_name
            if not ckpt_path.exists():
                raise RuntimeError(f'No checkpoint found at {ckpt_path}!')
        ckpt_name = ckpt_path.stem

        return ckpt_path, ckpt_name

    def get_run_path(self, wandb_version):
        config_path = Path(__file__).parent.parent.parent.parent / \
            'configs' / 'config.yaml'
        with config_path.open('r') as f:
            config = yaml.load(f)
            outputs_dir = Path(config['outputs_dir'])
        wandb_dir = outputs_dir / 'wandb'
        for run_dir in wandb_dir.iterdir():
            if wandb_version in run_dir.stem:
                print(f'run_dir = {run_dir}')
                return run_dir
        raise RuntimeError(
            f'run with wandb_version={wandb_version} not found; is {wandb_dir} correct?')

    def save_predictions(self, batch_size=512, num_workers=32):
        self._save_predictions(
            self.model, self.datamodule, self.run_prediction_dir, self.cfg, batch_size, num_workers)

    def _save_predictions(self, model, datamodule, run_prediction_dir, cfg, batch_size=512, num_workers=32):
        datamodule.num_workers = num_workers
        datamodule.batch_size = batch_size
        model.cuda(0)
        model.eval()

        for split, dataloader in zip(('test', 'val', 'train'), (datamodule.test_dataloader(), datamodule.val_dataloader(), datamodule.train_dataloader())):
            if split != 'val':
                continue
            output_dir = run_prediction_dir / split
            output_dir.mkdir(exist_ok=True, parents=True)
            with torch.no_grad():
                for i, batch in tqdm(enumerate(dataloader), total=math.ceil(len(dataloader.dataset.files) / datamodule.batch_size)):
                    features = batch['features'].to(model.device)
                    inverse_map = batch['inverse_map'].F.type(torch.long)
                    subbatch_idx = features.C[..., -1]
                    subbatch_im_idx = batch['inverse_map'].C[..., -
                                                             1].to(model.device)
                    if cfg.criterion.task == 'instance':
                        embedding = model(features).cpu().numpy()
                    elif cfg.criterion.task == 'semantic':
                        labels = torch.argmax(
                            model(features), dim=1).cpu().numpy()
                    elif cfg.criterion.task == 'panoptic':
                        out_c, out_e = model(features)
                        labels = torch.argmax(out_c, dim=1).cpu().numpy()
                        embedding = out_e.cpu().numpy()
                    for j in torch.unique(subbatch_idx):
                        event_path = dataloader.dataset.files[i *
                                                               datamodule.batch_size + int(j.cpu().numpy())]
                        event_name = event_path.stem
                        output_path = output_dir / event_name
                        mask = (subbatch_idx == j).cpu().numpy()
                        im_mask = (subbatch_im_idx == j).cpu().numpy()
                        if cfg.criterion.task == 'instance':
                            np.savez_compressed(
                                output_path, embedding=embedding[mask][inverse_map[im_mask]])
                        elif cfg.criterion.task == 'semantic':
                            np.savez_compressed(
                                output_path, labels=labels[mask][inverse_map[im_mask]])
                        elif cfg.criterion.task == 'panoptic':
                            np.savez_compressed(
                                output_path, labels=labels[mask][inverse_map[im_mask]], embedding=embedding[mask][inverse_map[im_mask]])


    def get_events(self, split, n=-1, batch_size=512):
        datamodule = self.datamodule

        if split == 'train':
            dataloader = datamodule.train_dataloader()
        elif split == 'val':
            dataloader = datamodule.val_dataloader()
        elif split == 'test':
            dataloader = datamodule.test_dataloader()
        else:
            raise NotImplementedError()

        if n == -1:
            input_paths = dataloader.dataset.files
        else:
            m = len(dataloader.dataset.files)
            input_paths = dataloader.dataset.files[m-n:m]

        events = []

        pred_dir = self.run_prediction_dir / split
        if len([f for f in pred_dir.glob('*.npz')]) == 0:
            self.save_predictions(batch_size=batch_size)

        logging.info('Loading events...')
        for input_path in tqdm(input_paths):
            event_name = input_path.stem
            pred_path = pred_dir / f'{event_name}.npz'
            events.append(self.make_event(
                input_path, pred_path))
        return events

    def make_event(self, input_path, pred_path):
        return BaseEvent(input_path, pred_path, self.task)

    def load_model(self, ckpt_path):
        model_str = self.cfg.model.target._target_
        comps = model_str.split('.')
        module = importlib.import_module('.'.join(comps[:-1]))
        cls = getattr(module, comps[-1])
        return cls.load_from_checkpoint(str(ckpt_path))