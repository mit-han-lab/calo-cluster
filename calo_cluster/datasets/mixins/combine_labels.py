from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .base import AbstractBaseDataset


@dataclass
class CombineLabelsMixin(AbstractBaseDataset):
    "Combines labels into single 'label' field."
    task: str

    def _get_numpy(self, index: int) -> Dict[str, Any]:
        return_dict = super()._get_numpy()

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
