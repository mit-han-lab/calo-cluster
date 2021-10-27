# %%
from logging import root
import uproot
from tqdm.auto import tqdm
import multiprocessing as mp
import pandas as pd
from functools import partial
import plotly.express as px
# %%
uproot_file = uproot.open('/data/vertex/vertexperformance_AMVF.root')
# %%
uproot_file.keys()
# %%
truth_tree = uproot_file['Truth_Vertex_PV_Selected;6']
reco_tree = uproot_file['Reco_Vertex;4']
# %%
def get_jagged(tree, prefix):
    jagged_dict = {}
    for k,v in tqdm(tree.items()):
        if not k.startswith(prefix):
            continue
        jagged_dict[k[len(prefix):]] = v.array()
    return jagged_dict

def process_truth(n):
        df_dict = {k: truth_dict[k][n] for k in truth_dict.keys()}
        flat_event = pd.DataFrame(df_dict)
        return flat_event

def process_reco(n):
        df_dict = {k: reco_dict[k][n] for k in reco_dict.keys()}
        flat_event = pd.DataFrame(df_dict)
        return flat_event

def get_events(tree, process):
    events = []
    ncpus = 8
    event_indices = range(len(tree[0].array()))
    with mp.Pool(ncpus) as p:
        with tqdm(total=len(event_indices)) as pbar:
            for r in p.imap_unordered(process, event_indices):
                events.append(r)
                pbar.update()
    return events

# %%
truth_dict = get_jagged(truth_tree, prefix='truth_vtx_fitted_trk_')
# %%
reco_dict = get_jagged(reco_tree, prefix='reco_vtx_fitted_trk_')
# %%
truth_events = get_events(truth_tree, process=process_truth)

# %%
reco_events = get_events(reco_tree, process=process_reco)
# %%
truth_events[0].columns
# %%
reco_events[0].columns

# %%
coords = ['d0', 'z0', 'phi', 'theta', 'qp']
flat_df = pd.concat(e[coords] for e in reco_events)
# %%
flat_df
# %%
scale = flat_df.max() - flat_df.min()
# %%
scale[0] = 0.05
print(scale)
# %%
plot_df = reco_events[0].copy()
plot_df['vtxID'] = plot_df['vtxID'].astype(str)

coords = ['d0', 'z0', 'phi', 'theta', 'qp']
plot_df[coords] /= scale

import plotly.express as px
px.scatter(plot_df, x='qp', y='z0', color='vtxID')
# %%
px.histogram(flat_df.loc[:5000, 'qp'])
# %%
reco_events[0].columns
# %%
from pathlib import Path
from calo_cluster.datasets.vertex import VertexDataModule
root_data_path = Path('/data/vertex')
raw_data_dir = root_data_path / 'raw'
#VertexDataModule.root_to_pickle(root_data_path, raw_data_dir)
# %%
occupancy = VertexDataModule.voxel_occupancy(voxel_size=0.001, dataset='vertex')
# %%
px.histogram(occupancy)
# %%
