import math
from pathlib import Path
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
from hgcal_dev.evaluation.metrics.instance import PanopticQuality
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
        event_prediction = np.load(self.pred_path)
        if self.task == 'panoptic':
            self.embedding = event_prediction['embedding']
            self.pred_class_labels = event_prediction['labels']
        elif self.task == 'instance':
            self.embedding = event_prediction['embedding']
        elif self.task == 'semantic':
            self.pred_class_labels = event_prediction['labels']
        self.input_event = input_event

    @property
    def pred_instance_labels(self):
        if self._pred_instance_labels is None:
            self._pred_instance_labels = self.clusterer.cluster(self.embedding)
        return self._pred_instance_labels

    def pq(self, use_weights=False):
        assert self.task == 'panoptic' or self.task == 'instance'
        if self.task == 'panoptic':
            pq_metric = PanopticQuality(num_classes=self.num_classes)

            outputs = (self.pred_class_labels, self.pred_instance_labels)
            targets = (self.input_event[self.class_label].values,
                       self.input_event[self.instance_label].values)
        elif self.task == 'instance':
            pq_metric = PanopticQuality(semantic=False)

            outputs = self.pred_instance_labels
            targets = self.input_event[self.instance_label].values

        if use_weights:
            if self.weight_name is None:
                raise RuntimeError('No weight name given!')
            weights = self.input_event[self.weight_name]
        else:
            weights = None
        pq_metric.add(outputs, targets, weights=weights)

        return pq_metric.compute()


class BaseExperiment():
    def __init__(self, wandb_version, ckpt_name=None):
        self.wandb_version = wandb_version
        run_path = self.get_run_path(wandb_version)
        self.run_path = run_path
        cfg_path = run_path / 'files' / '.hydra' / 'config.yaml'
        self.cfg = OmegaConf.load(cfg_path)
        ckpt_dir = Path(self.cfg.outputs_dir) / self.cfg.wandb.project / \
            self.cfg.wandb.version / 'checkpoints'
        if ckpt_name is None:
            ckpt_path = ckpt_dir / 'last.ckpt'
            if not ckpt_path.exists():
                ckpt_path = [p for p in sorted(ckpt_dir.glob('*.ckpt'))][-1]
            print(f'no checkpoint name given, using {ckpt_path}.')
        else:
            ckpt_path = ckpt_dir / ckpt_name
            if not ckpt_path.exists():
                raise RuntimeError(f'No checkpoint found at {ckpt_path}!')
        self.ckpt_name = ckpt_path.stem
        self.run_prediction_dir = Path(
            self.cfg.predictions_dir) / self.cfg.wandb.version / self.ckpt_name
        self.run_prediction_dir.mkdir(exist_ok=True, parents=True)
        self.model = SPVCNN.load_from_checkpoint(str(ckpt_path))

        self.datamodule = hydra.utils.instantiate(self.cfg.dataset)
        self.datamodule.batch_size = 1
        self.datamodule.prepare_data()
        self.datamodule.setup(stage=None)

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

    def save_predictions(self):
        model = self.model
        datamodule = self.datamodule
        datamodule.num_workers = 32
        model.cuda(0)
        model.eval()
        datamodule.batch_size = 512
        datamodule.prepare_data()
        datamodule.setup(stage=None)
        for split, dataloader in zip(('test', 'val', 'train'), (datamodule.test_dataloader(), datamodule.val_dataloader(), datamodule.train_dataloader())):
            output_dir = self.run_prediction_dir / split
            output_dir.mkdir(exist_ok=True, parents=True)
            with torch.no_grad():
                for i, batch in tqdm(enumerate(dataloader), total=math.ceil(len(dataloader.dataset.events) / datamodule.batch_size)):
                    features = batch['features'].to(model.device)
                    inverse_map = batch['inverse_map'].F.type(torch.long)
                    subbatch_idx = features.C[..., -1]
                    subbatch_im_idx = batch['inverse_map'].C[..., -1].to(model.device)
                    if self.cfg.criterion.task == 'instance':
                        embedding = model(features).cpu().numpy()
                    elif self.cfg.criterion.task == 'semantic':
                        labels = torch.argmax(model(features), dim=1).cpu().numpy()
                    elif self.cfg.criterion.task == 'panoptic':
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
                        if self.cfg.criterion.task == 'instance':
                            np.savez_compressed(
                                output_path, embedding=embedding[mask][inverse_map[im_mask]])
                        elif self.cfg.criterion.task == 'semantic':
                            np.savez_compressed(
                                output_path, labels=labels[mask][inverse_map[im_mask]])
                        elif self.cfg.criterion.task == 'panoptic':
                            np.savez_compressed(
                                output_path, labels=labels[mask][inverse_map[im_mask]], embedding=embedding[mask][inverse_map[im_mask]])

    def loss(self, split):
        task = self.cfg.dataset.task
        assert task in ('instance', 'semantic', 'panoptic')
        if task == 'instance':
            embed_criterion = hydra.utils.instantiate(
                self.cfg.criterion.embed)
            ret = {'loss': 0.0}
        elif task == 'panoptic':
            semantic_criterion = hydra.utils.instantiate(
                self.cfg.criterion.semantic)
            embed_criterion = hydra.utils.instantiate(
                self.cfg.criterion.embed)
            ret = {'loss': 0.0, 'class_loss': 0.0, 'embed_loss': 0.0}
        elif task == 'semantic':
            semantic_criterion = hydra.utils.instantiate(
                self.cfg.criterion.semantic)
            ret = {'loss': 0.0}

        model = self.model
        datamodule = self.datamodule
        datamodule.num_workers = 32
        model.cuda(0)
        model.eval()
        datamodule.batch_size = 512
        datamodule.prepare_data()
        datamodule.setup(stage=None)

        if split == 'train':
            dataloader = datamodule.train_dataloader()
        elif split == 'val':
            dataloader = datamodule.val_dataloader()
        elif split == 'test':
            dataloader = datamodule.test_dataloader()
        n_batches = math.ceil(
            len(dataloader.dataset.events) / datamodule.batch_size)

        with torch.no_grad():
            for batch in tqdm(dataloader, total=n_batches):
                features = batch['features'].to(model.device)
                targets = batch['labels'].F.long().to(model.device)
                outputs = model(features)
                subbatch_indices = features.C[..., -1]
                if 'weights' in batch:
                    weights = batch['weights'].F.to(model.device)
                else:
                    weights = None
                if task == 'semantic':
                    loss = semantic_criterion(outputs, targets)
                    ret['loss'] += loss / n_batches
                elif task == 'instance':
                    loss = embed_criterion(
                        outputs, targets, subbatch_indices, weights)
                    ret['loss'] += loss / n_batches
                elif task == 'panoptic':
                    class_loss = semantic_criterion(outputs[0], targets[:, 0])
                    embed_loss = embed_criterion(
                        outputs[1], targets[:, 1], subbatch_indices, weights)
                    loss = class_loss + self.cfg.criterion.alpha * embed_loss
                    ret['loss'] += loss / n_batches
                    ret['class_loss'] += class_loss / n_batches
                    ret['embed_loss'] += embed_loss / n_batches
        return ret

    def get_events(self, split, n=-1):
        pred_dir = self.run_prediction_dir / split
        if len([f for f in pred_dir.glob('*.npz')]) == 0:
            self.save_predictions()

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
            event_name = input_path.stem
            pred_path = pred_dir / f'{event_name}.npz'
            events.append(self.make_event(
                input_path, pred_path, task=self.cfg.criterion.task))
        return events

    def make_event(input_path, pred_path, task):
        raise NotImplementedError()
