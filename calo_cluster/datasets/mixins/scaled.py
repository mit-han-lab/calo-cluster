from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np

from .base import AbstractBaseDataModule, AbstractBaseDataset


@dataclass
class ScaledDatasetMixin(AbstractBaseDataset):
    """Dataset mixin to optionally transform features and/or coordinates.
    
    Parameters:
    transform_features -- if true, use scaling on the features (x = (x - features_loc) / features_scale)
    transform_coords -- same as transform_features, but for coords
    """
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

@dataclass
class ScaledDataModuleMixin(AbstractBaseDataModule):
    transform_features: bool
    features_loc: Union[List[float], None]
    features_scale: Union[List[float], None]

    transform_coords: bool
    coords_loc: Union[List[float], None]
    coords_scale: Union[List[float], None]

    def make_dataset_kwargs(self) -> Dict[str, Any]:
        kwargs = {
            'transform_features': self.transform_features,
            'features_loc': self.features_loc,
            'features_scale': self.features_scale,
            'transform_coords': self.transform_coords,
            'coords_loc': self.coords_loc,
            'coords_scale': self.coords_scale
        }
        kwargs.update(super().make_dataset_kwargs())
        return kwargs