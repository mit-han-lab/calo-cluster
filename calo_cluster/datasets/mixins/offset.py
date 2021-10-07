from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

from .base import AbstractBaseDataset


@dataclass
class OffsetDatasetMixin(AbstractBaseDataset):
    "A dataset that supports cluster offsets."
    valid_labels: list

    @abstractmethod
    def _get_offset(self, coordinates, semantic_labels, instance_labels):
        pass

    def _get_numpy(self, index: int) -> Dict[str, Any]:
        return_dict = super()._get_numpy()
        return_dict['offsets'] = self._get_offset(
            return_dict['coordinates'], return_dict['semantic_labels'], return_dict['instance_labels'])
        return return_dict
