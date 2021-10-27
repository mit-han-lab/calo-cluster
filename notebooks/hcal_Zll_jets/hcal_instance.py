# %%
from calo_cluster.evaluation.experiments.base_experiment import BaseExperiment
from pathlib import Path 
from calo_cluster.clustering.meanshift import MeanShift
from tqdm.auto import tqdm
# %%
wandb_version = '1fzko8ui'
exp = BaseExperiment(wandb_version)

# %%
evts = exp.get_events('val')

# %%
clusterer = MeanShift(bandwidth=0.02, use_semantic=True, ignore_semantic_labels=(0,))
for evt in tqdm(evts):
    evt.pred_instance_labels = clusterer.cluster(embedding=evt.embedding, semantic_labels=evt.input_event['pf_hit'].to_numpy())
    
# %%
from calo_cluster.evaluation.metrics.instance import PanopticQuality
# %%
evts[0].input_event
# %%
plot_df = evts[0].input_event
plot_df['pred_label'] = evts[0].pred_instance_labels
plot_df['pred_label'] = plot_df['pred_label'].astype(str)
plot_df['PFcluster0Id'] = plot_df['PFcluster0Id'].astype(str)
import plotly.express as px
mask = plot_df['pf_hit'] != 0
px.scatter_3d(plot_df[mask], x='x', y='y', z='z', size='energy', color='pred_label')
# %%
px.scatter_3d(plot_df[mask], x='x', y='y', z='z', size='energy', color='PFcluster0Id')
# %%
pq = PanopticQuality(num_classes=2, semantic=False)
for evt in tqdm(evts):
    mask = evt.input_event['pf_hit'] != 0
    pq.add(evt.pred_instance_labels[mask], evt.input_event['PFcluster0Id'].astype(int).to_numpy()[mask])
print(f'pq = {pq.compute()}')
# %%
