# %%
from calo_cluster.datasets.hcal_Zll_jets import HCalZllJetsDataModule
from pathlib import Path


# %%
root_data_path = Path('/data/hcal_Zll_jets')
raw_data_dir = root_data_path / 'raw'

# %%
HCalZllJetsDataModule.root_to_pickle(root_data_path, raw_data_dir)
# %%
import uproot
from tqdm import tqdm
root_data = root_data_path / '27Apr_withPFcluster_sum_v2.root'
root_dir = uproot.open(root_data)
# %%
root_events = root_dir.get('Events;1')
# %%
root_events.keys()
# %%
root_events['RecHit'].keys()
# %%
import pandas as pd
df = pd.DataFrame()
data_dict = {}
for k, v in tqdm(root_events['RecHit'].items()):
    data_dict[k.split('.')[1]] = v.array()


# %%
n = 0
noise_id=-99
pf_noise_id = -1
df_dict = {k: data_dict[k][n] for k in data_dict.keys()}
flat_event = pd.DataFrame(df_dict)
pf_noise_mask = (flat_event['PFcluster0Id'] == pf_noise_id)
flat_event['pf_hit'] = 0
flat_event.loc[~pf_noise_mask, 'pf_hit'] = 1
flat_event.astype({'hit': int})
hit_mask = (flat_event['genE'] > 0.2)
noise_mask = ~hit_mask
flat_event['hit'] = hit_mask.astype(int)
flat_event.loc[noise_mask, 'trackId'] = noise_id
# %%
for n in tqdm(len(data_dict[list(data_dict.keys())[0]])):
    df_dict = {k: data_dict[k][n] for k in data_dict.keys()}
    flat_event = pd.DataFrame(df_dict)
    pf_noise_mask = (flat_event['PFcluster0Id'] == pf_noise_id)
    flat_event['pf_hit'] = 0
    flat_event.loc[~pf_noise_mask, 'pf_hit'] = 1
    flat_event.astype({'hit': int})
    hit_mask = (flat_event['genE'] > 0.2)
    noise_mask = ~hit_mask
    flat_event['hit'] = hit_mask.astype(int)
    flat_event.loc[noise_mask, 'trackId'] = noise_id
    flat_event.to_pickle(raw_data_dir / f'event_{n:05}.pkl')

# %%
import plotly.express as px

px.scatter_3d(flat_event, x='x', y='y', z='z', color='pf_hit', size='energy')
# %%
flat_event['PFcluster0Id']

# %%
