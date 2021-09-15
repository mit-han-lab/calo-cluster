# %%
from calo_cluster.datasets.semantic_kitti import SemanticKITTIDataModule
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import hydra
import plotly.express as px
# %%
with initialize_config_dir(config_dir='/home/alexj/hgcal-dev/configs'):
    cfg = compose(config_name='config', overrides=['dataset=semantic_kitti'])
    dm = hydra.utils.instantiate(cfg.dataset, task='panoptic')
# %%
dm.prepare_data()
dm.setup('fit')
dataset = dm.train_dataloader().dataset

# %%
data_dict = dataset.get_numpy(0)
# %%
data_dict
# %%
data_dict['labels'][:, 1]
# %%
import pandas as pd
plot_df = pd.DataFrame({'x': data_dict['features'][:, 0], 'y': data_dict['features'][:, 1], 'z': data_dict['features'][:, 2], 'color': data_dict['labels'][:, 1]})
plot_df['color'] = plot_df['color'].astype(str)
px.scatter_3d(plot_df, x='x', y='y', z='z', color='color')
# %%
data_dict['labels'][:, 1].shape
# %%
