# %%
from calo_cluster.datasets.simple import SimpleDataModule
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import hydra
import plotly.express as px
# %%
with initialize_config_dir(config_dir='/home/alexj/hgcal-dev/configs'):
    cfg = compose(config_name='config', overrides=['dataset=simple', '+task=panoptic'])
    dm = hydra.utils.instantiate(cfg.dataset, task='panoptic')
# %%
dm.prepare_data()
dm.setup('fit')
dataset = dm.train_dataloader().dataset
# %%
data_dict = dataset.get_numpy(0)
# %%
px.scatter_3d(x=data_dict['features'][:, 0], y=data_dict['features'][:, 1], z=data_dict['features'][:, 2])
# %%
sparse_dict = dataset[0]
px.scatter_3d(x=sparse_dict['features'].C[:, 0], y=sparse_dict['features'].C[:, 1], z=sparse_dict['features'].C[:, 2])

# %%
# plot instance labels
px.scatter_3d(x=sparse_dict['features'].C[:, 0], y=sparse_dict['features'].C[:, 1], z=sparse_dict['features'].C[:, 2], size=sparse_dict['features'].F[:, 3], color=sparse_dict['labels'].F[:, 1].astype(str))
# %%
# plot semantic labels
px.scatter_3d(x=sparse_dict['features'].C[:, 0], y=sparse_dict['features'].C[:, 1], z=sparse_dict['features'].C[:, 2], size=sparse_dict['features'].F[:, 3], color=sparse_dict['labels'].F[:, 0].astype(str))
# %%
