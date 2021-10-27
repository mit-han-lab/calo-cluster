# %%
from calo_cluster.evaluation.experiments.base_experiment import BaseExperiment
from pathlib import Path 
from calo_cluster.clustering.meanshift import MeanShift
from tqdm.auto import tqdm
from calo_cluster.evaluation.metrics.instance import PanopticQuality
# %%
# Load the trained model -- replace wandb_version as appropriate.
wandb_version = '3d4h8gwo'
exp = BaseExperiment(wandb_version)

# %%
# Load the validation events.
evts = exp.get_events('val')

# %%
# Run the mean shift clustering to get instance labels.
clusterer = MeanShift(bandwidth=0.05, use_semantic=True, ignore_semantic_labels=(0,))
for evt in tqdm(evts):
    evt.pred_instance_labels = clusterer.cluster(embedding=evt.embedding, semantic_labels=evt.input_event['semantic_label'].to_numpy())
    
# %%
# Plot the predicted and true data.
plot_df = evts[0].input_event
plot_df['pred_label'] = evts[0].pred_instance_labels
plot_df['pred_label'] = plot_df['pred_label'].astype(str)
plot_df['instance_label'] = plot_df['instance_label'].astype(str)
import plotly.express as px
px.scatter_3d(plot_df, x='x', y='y', z='z', size='E', color='pred_label')
# %%
px.scatter_3d(plot_df, x='x', y='y', z='z', size='E', color='instance_label')
# %%
# Calculate the panoptic quality, ignoring the noise.
pq = PanopticQuality(num_classes=2, semantic=False)
for evt in tqdm(evts):
    mask = evt.input_event['semantic_label'] != 0
    pq.add(evt.pred_instance_labels[mask], evt.input_event['instance_label'].astype(int).to_numpy()[mask])
print(f'pq = {pq.compute()}')
