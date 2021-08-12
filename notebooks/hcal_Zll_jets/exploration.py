# %%
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import hydra
import plotly.express as px
# %%
with initialize_config_dir(config_dir='/home/alexj/calo-cluster/configs'):
    cfg = compose(config_name='config', overrides=['dataset=hcal_Zll_jets'])
    dm = hydra.utils.instantiate(cfg.dataset, task='panoptic')
# %%
dm.prepare_data()
dm.setup('fit')
dataset = dm.train_dataloader().dataset
# %%
dataset[0]
# %%
n = 2
block = dataset[n]['features'].F
labels = dataset[n]['labels'].F
df_dict = {'x': block[:, 0], 'y': block[:, 1], 'z': block[:, 2], 't': block[:, 3], 'E': block[:, 4]}
df_dict['semantic_label'] = labels[:, 0]
df_dict['instance_label'] = labels[:, 1]
# %%
import pandas as pd
df = pd.DataFrame(df_dict)
df['semantic_label'] = df['semantic_label'].astype(str)
df['instance_label'] = df['instance_label'].astype(str)
px.scatter_3d(df, x='x', y='y', z='z', size='E', color='instance_label')
# %%
