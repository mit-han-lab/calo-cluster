# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from hgcal_dev.evaluation.studies.clustering_study import ClusteringStudy
from hgcal_dev.evaluation.experiments.antikt_hcal_experiment import AntiKtHCalEvent, AntiKtHCalExperiment
from hgcal_dev.evaluation.experiments.hcal_experiment import HCalExperiment
from hgcal_dev.evaluation.studies.panoptic_study import PanopticStudy
from hgcal_dev.clustering.meanshift import MeanShift
from hgcal_dev.clustering.identity_clusterer import IdentityClusterer
from hgcal_dev.datasets.hcal import HCalDataModule
import pandas as pd
# %%
exp = HCalExperiment(wandb_version=('3ajhmqom', '2vq4h349'))


# %%
ps = PanopticStudy(exp)

# %%
clusterer_factory = lambda bw: MeanShift(bandwidth=bw, use_semantic=True, ignore_class_label=0)
ps.bandwidth_study(clusterer_factory, nevents=100, stepsize=0.05, bounds=(0.001, 0.15), T=0.1, x0=0.02, ignore_class_labels=(0,))

# %%
ps.confusion_matrix()

# %%
ps.clusterer = MeanShift(bandwidth=0.02, use_semantic=True, ignore_class_label=0)
ps.pq(nevents=100, use_weights=True, ignore_class_labels=(0,))

# %%
ps.clusterer = MeanShift(bandwidth=0.022, use_semantic=False)
ps.pq(use_weights=True)

# %%
ps.clusterer = MeanShift(bandwidth=0.022, use_semantic=True, ignore_class_label=0)
ps.qualitative_cluster_study(n=3)

# %%
ps.clusterer = MeanShift(bandwidth=0.05, use_semantic=True, ignore_class_label=0)
data_dict = ps.response(nevents=-1, splits=('val',), nbins=100, lo=0.5, hi=100, match_highest=True)

# %%
import numpy as np
from pathlib import Path
import plotly.express as px
from hgcal_dev.evaluation.studies import functional as F
cluster_df = data_dict['val']['cluster_dfs'][1]
bin_edges = [a for a in np.linspace(0.5, 15, 15)] + [a for a in np.linspace(16, 100, 8)]
means, errors = F.make_bins(
    cluster_df['energy'], cluster_df['energy response'], bin_edges=bin_edges)
bin_df = pd.DataFrame(
    {'response': means, 'error': errors, 'energy': bin_edges})
fig = px.scatter(bin_df, x='energy',
                    y='response', error_y='error')
out_path = Path('/home/alexj/plots/3ajhmqom_2vq4h349/last_last/match_highest') / \
    f'val_class_1_binned_energy_response_histogram_custom_bins.png'
fig.write_image(str(out_path), scale=10)
# %%
occupancies = ps.voxel_occupancy()

# %%
df = pd.DataFrame({'voxel occupancy': occupancies})
fig = px.histogram(df, x='voxel occupancy')
out_path = Path('/home/alexj/plots/3ajhmqom_2vq4h349/last_last') / 'voxel_occupancies.png'
fig.write_image(str(out_path), scale=10)

# %%
exp.get_events('val', n=1)[0].input_event.columns
# %%
exp_antikt = AntiKtHCalExperiment(wandb_version=('3ajhmqom', '2vq4h349'))
# %%
ps_antikt = ClusteringStudy(exp_antikt, clusterer=IdentityClusterer(use_semantic=False))

# %%
ps_antikt.pq(nevents=-1, ignore_class_labels=(0,))

# %%
exp.get_events('val', n=2)[1].input_event
# %%
antikt_data_dict = ps_antikt.response(nevents=-1, nbins=100, hi=100, lo=0.5, out_dir='antikt', match_highest=True)
# %%
