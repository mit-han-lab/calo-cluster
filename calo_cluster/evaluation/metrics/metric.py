from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Union

import torchmetrics
from torchmetrics.metric import Metric

from calo_cluster.evaluation.experiments.base_experiment import BaseExperiment



@dataclass
class TestMetric:
    experiment: BaseExperiment
    metric: Metric
    save_name: Union[str, Path, None]

    def add_from_dict(self, subbatch):
        """Add data from subbatch dict."""
        self.metric()
    
    @abstractmethod
    def _save(self, path):
        """Calculate (and possibly save) metric data."""
        raise NotImplementedError()

    def save(self):
        path = self.get_path()
        if path is not None:
            logging.info(f'writing metric data to {path}')
        self._save(path)

    def get_path(self):
        if self.save_name is not None:
            path = (self.experiment.plots_dir / self.save_name)
            path.mkdir(exist_ok=True, parents=True)
            return path
        else:
            return None
    