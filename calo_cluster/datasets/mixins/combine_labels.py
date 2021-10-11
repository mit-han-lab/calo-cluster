from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .base import AbstractBaseDataModule, AbstractBaseDataset


@dataclass
class CombineLabelsDatasetMixin(AbstractBaseDataset):
    """Combines labels into single 'label' field.
    
    Parameters:
    task -- the type of ML task that will be performed on this dataset (semantic, instance, panoptic)"""
    task: str

    def _get_numpy(self, index: int) -> Dict[str, Any]:
        return_dict = super()._get_numpy(index)

        semantic_labels = return_dict.pop('semantic_labels')
        instance_labels = return_dict.pop('instance_labels')

        if self.task == 'panoptic':
            return_dict['labels'] = np.stack(
                (semantic_labels, instance_labels), axis=-1)
        elif self.task == 'semantic':
            return_dict['labels'] = semantic_labels
        elif self.task == 'instance':
            return_dict['labels'] = instance_labels
        else:
            raise RuntimeError(f'Unknown task = "{self.task}"')

        return return_dict

@dataclass
class CombineLabelsDataModuleMixin(AbstractBaseDataModule):
    task: str

    def make_dataset_kwargs(self) -> Dict[str, Any]:
        kwargs = {
            'task': self.task
        }
        kwargs.update(super().make_dataset_kwargs())
        return kwargs