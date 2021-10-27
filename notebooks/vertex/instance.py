# %%
from calo_cluster.evaluation.experiments.base_experiment import BaseExperiment
from pathlib import Path 
from calo_cluster.clustering.meanshift import MeanShift
from tqdm.auto import tqdm
from calo_cluster.evaluation.metrics.instance import PanopticQuality
# %%
# Load the trained model -- replace wandb_version as appropriate.
wandb_version = '1ll9d8wi'
exp = BaseExperiment(wandb_version)

# %%
# Load the validation events.
evts = exp.get_events('val')

# %%
# Run the mean shift clustering to get instance labels.
clusterer = MeanShift(bandwidth=0.05, use_semantic=False)
for evt in tqdm(evts):
    evt.pred_instance_labels = clusterer.cluster(embedding=evt.embedding)
    
# %%
# Plot the predicted and true data.
plot_df = evts[0].input_event
plot_df['pred_label'] = evts[0].pred_instance_labels
plot_df['pred_label'] = plot_df['pred_label'].astype(str)
plot_df['instance_label'] = plot_df['reco_vtxID'].astype(str)
import plotly.express as px
px.scatter_3d(plot_df, x='d0', y='z0', z='qp', color='pred_label')
# %%
px.scatter_3d(plot_df, x='d0', y='z0', z='qp', color='instance_label')
# %%
# Calculate the panoptic quality, ignoring the noise.
pq = PanopticQuality(num_classes=2, semantic=False)
for evt in tqdm(evts):
    pq.add(evt.pred_instance_labels, evt.input_event['reco_vtxID'].astype(int).to_numpy())
print(f'pq = {pq.compute()}')

# %%
