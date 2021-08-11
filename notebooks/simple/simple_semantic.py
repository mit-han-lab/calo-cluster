# %%
from calo_cluster.evaluation.experiments.base_experiment import BaseExperiment
from pathlib import Path 
# %%
wandb_version = 'bo6242ff'
ckpt_name = '5-2999.ckpt'
exp = BaseExperiment(wandb_version, ckpt_name=ckpt_name)

# %%
evts = exp.get_events('val', n=1000)

# %%
from calo_cluster.evaluation.metrics.classification import mIoU
# %%
iou = mIoU(evts, num_classes=2, semantic_label='semantic_label')
# %%
plot_df = evts[0].input_event
plot_df['pred_label'] = evts[0].pred_semantic_labels
plot_df['pred_label'] = plot_df['pred_label'].astype(str)
plot_df['semantic_label'] = plot_df['semantic_label'].astype(str)
import plotly.express as px
px.scatter_3d(plot_df, x='x', y='y', z='z', size='E', color='pred_label')
# %%
px.scatter_3d(plot_df, x='x', y='y', z='z', size='E', color='semantic_label')
# %%
