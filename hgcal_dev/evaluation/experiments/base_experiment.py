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
from hgcal_dev.clustering.meanshift import MeanShift
from hgcal_dev.models.spvcnn import SPVCNN
from omegaconf import OmegaConf
from plotly.subplots import make_subplots
from sklearn.metrics import auc, confusion_matrix, roc_curve
from tqdm import tqdm


class BaseEvent():
    """Aggregates the input/output/clustering for a single event."""

    def __init__(self, input_path, class_label: str = None, instance_label: str = None, pred_path: Path = None, task: str = 'panoptic', clusterer=MeanShift(), num_classes: int = 2, weight_name: str = None):
        self.input_path = input_path
        self.pred_path = pred_path
        self.task = task
        self.class_label = class_label
        self.instance_label = instance_label
        self._pred_instance_labels = None
        self.num_classes = num_classes
        self.clusterer = clusterer
        self.weight_name = weight_name
        self._load()

    def _load(self):
        input_event = pd.read_pickle(self.input_path)
        self.input_event = input_event

        self.multiple_models = not issubclass(type(self.pred_path), Path) and type(self.pred_path) != str

        if self.multiple_models:
            event_prediction = {}
            event_prediction['labels'] = np.load(self.pred_path[0])['labels']
            event_prediction['embedding'] = np.load(
                self.pred_path[1])['embedding']
        else:
            event_prediction = np.load(self.pred_path)

        if self.task == 'panoptic':
            self.pred_class_labels = event_prediction['labels']
            self.embedding = event_prediction['embedding']
        elif self.task == 'instance':
            self.embedding = event_prediction['embedding']
        elif self.task == 'semantic':
            self.pred_class_labels = event_prediction['labels']


class BaseExperiment():
    def __init__(self, wandb_version, ckpt_name=None):
        ''' If separate semantic and instance segmentation models are used, wandb_version and ckpt_name can be lists. 
            The first element should be the semantic model, while the second should be the instance model.'''

        self.multiple_models = type(wandb_version) != str
        self.wandb_version = wandb_version
            
        if self.multiple_models:
            assert ckpt_name is None or len(wandb_version) == len(ckpt_name)
            self.run_path = [self.get_run_path(v) for v in self.wandb_version]
            cfg_path = [p / 'files' / '.hydra' /
                        'config.yaml' for p in self.run_path]
            self.cfg = [OmegaConf.load(p) for p in cfg_path]
            if ckpt_name is None:
                ckpt_name = [None] * len(self.cfg)
            ckpts = [self._get_ckpt(cfg, name)
                     for cfg, name in zip(self.cfg, ckpt_name)]
            self.ckpt_path = [p for p, _ in ckpts]
            self.ckpt_name = [name for _, name in ckpts]
            self.run_prediction_dir = [Path(
                cfg.predictions_dir) / cfg.wandb.version / name for cfg, name in zip(self.cfg, self.ckpt_name)]
            for run_prediction_dir in self.run_prediction_dir:
                run_prediction_dir.mkdir(exist_ok=True, parents=True)
            self.model = [SPVCNN.load_from_checkpoint(
                str(p)) for p in self.ckpt_path]

            plots_dir = Path(self.cfg[1].plots_dir)
            self.plots_dir = plots_dir / '_'.join(self.wandb_version) / '_'.join(self.ckpt_name)

            self.datamodule = [hydra.utils.instantiate(cfg.dataset) for cfg in self.cfg]
            for dm in self.datamodule:
                dm.prepare_data()
                dm.setup(stage=None)
        else:
            self.run_path = self.get_run_path(self.wandb_version)
            cfg_path = self.run_path / 'files' / '.hydra' / 'config.yaml'
            self.cfg = OmegaConf.load(cfg_path)
            self.ckpt_path, self.ckpt_name = self._get_ckpt(
                self.cfg, ckpt_name)
            self.run_prediction_dir = Path(
                self.cfg.predictions_dir) / self.cfg.wandb.version / self.ckpt_name
            self.run_prediction_dir.mkdir(exist_ok=True, parents=True)
            self.model = SPVCNN.load_from_checkpoint(str(self.ckpt_path))

            plots_dir = Path(self.cfg.plots_dir)
            self.plots_dir = plots_dir / self.wandb_version / self.ckpt_name

            self.datamodule = hydra.utils.instantiate(self.cfg.dataset)
            self.datamodule.prepare_data()
            self.datamodule.setup(stage=None)

        self.plots_dir.mkdir(exist_ok=True, parents=True)

        if self.multiple_models:
            self.task = 'panoptic'
        else:
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
                return run_dir
        raise RuntimeError(
            f'run with wandb_version={wandb_version} not found; is {wandb_dir} correct?')

    def save_predictions(self, batch_size=512, ignore_model_idxs=None):
        if self.multiple_models:
            for i, (model, datamodule, run_prediction_dir, cfg) in enumerate(zip(self.model, self.datamodule, self.run_prediction_dir, self.cfg)):
                if i in ignore_model_idxs:
                    continue
                self._save_predictions(
                    model, datamodule, run_prediction_dir, cfg, batch_size)
        else:
            self._save_predictions(
                self.model, self.datamodule, self.run_prediction_dir, self.cfg, batch_size)

    def _save_predictions(self, model, datamodule, run_prediction_dir, cfg, batch_size=512):
        datamodule.num_workers = 32
        datamodule.batch_size = batch_size
        model.cuda(0)
        model.eval()

        for split, dataloader in zip(('test', 'val', 'train'), (datamodule.test_dataloader(), datamodule.val_dataloader(), datamodule.train_dataloader())):
            output_dir = run_prediction_dir / split
            output_dir.mkdir(exist_ok=True, parents=True)
            with torch.no_grad():
                for i, batch in tqdm(enumerate(dataloader), total=math.ceil(len(dataloader.dataset.events) / datamodule.batch_size)):
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
                        event_path = dataloader.dataset.events[i *
                                                               datamodule.batch_size + j]
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


    def get_events(self, split, n=-1):
        if self.multiple_models:
            datamodule = self.datamodule[1]
        else:
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
            input_paths = dataloader.dataset.events
        else:
            input_paths = dataloader.dataset.events[:n]

        events = []

        if self.multiple_models:
            pred_dir = [d / split for d in self.run_prediction_dir]
            ignore_model_idxs = []
            for i, d in enumerate(pred_dir):
                if len([f for f in d.glob('*.npz')]) != 0:
                    ignore_model_idxs.append(i)
            if len(ignore_model_idxs) != len(pred_dir):
                self.save_predictions(ignore_model_idxs=ignore_model_idxs)
        else:
            pred_dir = self.run_prediction_dir / split
            if len([f for f in pred_dir.glob('*.npz')]) == 0:
                self.save_predictions()

        logging.info('Loading events...')
        for input_path in tqdm(input_paths):
            event_name = input_path.stem
            if self.multiple_models:
                pred_path = [d / f'{event_name}.npz' for d in pred_dir]
            else:
                pred_path = pred_dir / f'{event_name}.npz'
            events.append(self.make_event(
                input_path, pred_path))
        return events

    def make_event(input_path, pred_path):
        raise NotImplementedError()
