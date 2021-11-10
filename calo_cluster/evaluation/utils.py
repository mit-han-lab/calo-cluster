from pathlib import Path
import sys
from typing import Union
import colorcet as cc
import numpy as np

def get_palette(instances: Union[np.array, int]):
    if type(instances) is int:
        size = instances
    else:
        size = np.unique(instances).shape[0]
    hex_colors = cc.glasbey_dark[:size]
    return [f'rgb{int(h[1:3], base=16),int(h[3:5], base=16),int(h[5:7], base=16)}' for h in hex_colors]