from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from .base import AbstractBaseDataset


@dataclass
class ScaledDatasetMixin(AbstractBaseDataset):
    transform_features: bool
    features_loc: List[float]
    features_scale: List[float]

    transform_coords: bool
    coords_loc: List[float]
    coords_scale: List[float]

    def _get_numpy(self, index: int) -> Dict[str, Any]:
        """Simple wrapper for _get_numpy that scales the features/coords."""
        return_dict = super()._get_numpy(index)

        if self.transform_features:
            return_dict['features'] = (return_dict['features'] - np.array(self.features_loc)) / \
                np.array(self.features_scale)

        if self.transform_coords:
            return_dict['coordinates'] = (return_dict['coordinates'] - np.array(self.coords_loc)
                                          ) / np.array(self.coords_scale)

        return return_dict
