# %%
from calo_cluster.evaluation.experiments.base_experiment import BaseExperiment
from calo_cluster.evaluation.experiments.base_offset_experiment import BaseOffsetExperiment
from pathlib import Path 
from calo_cluster.clustering.meanshift import MeanShift
from tqdm.auto import tqdm
from calo_cluster.evaluation.metrics.instance import PanopticQuality
import numpy as np
# %%
sem_wandb_version = '3bi8h2ur'
inst_wandb_version = '1zg4kakq'
sem_exp = BaseExperiment(sem_wandb_version, ckpt_name='5-5499.ckpt')
inst_exp = BaseOffsetExperiment(inst_wandb_version)
# %%
#sem_exp.save_predictions(batch_size=128, num_workers=32)
inst_exp.save_predictions(batch_size=128, num_workers=32)
# %%
sem_evts = sem_exp.get_events('val', n=1000)
inst_evts = inst_exp.get_events('val', n=1000)

# %%
def fix_instance_labels(pred_instance_labels, semantic_labels, labels_to_fix):
    fixed_labels = pred_instance_labels.copy()
    unique_labels = np.unique(semantic_labels)
    for l in unique_labels:
        if l not in labels_to_fix:
            continue
        mask = semantic_labels == l
        fixed_labels[mask] = 0
    return fixed_labels

# %%
sem_evts[0].input_event.columns

# %%
# true semantic labels, pred instance labels from coordinates
from tqdm.auto import tqdm
bws = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
#bws = [0.2]
rets = {}
for bw in bws:
    print(f'bw = {bw}')
    clusterer = MeanShift(bandwidth=bw, use_semantic=True, ignore_semantic_labels=(0,))
    pq = PanopticQuality(num_classes=2, ignore_index=None, ignore_semantic_labels=(0,))
    for s_evt, i_evt in tqdm(zip(sem_evts, inst_evts)):
        s_evt.pred_instance_labels = clusterer.cluster(embedding=i_evt.embedding, semantic_labels=s_evt.pred_semantic_labels)
        s_evt.pred_instance_labels = fix_instance_labels(s_evt.pred_instance_labels, s_evt.pred_semantic_labels, labels_to_fix=np.array([0]))
        pq.add((s_evt.pred_semantic_labels, s_evt.pred_instance_labels), (s_evt.input_event['semantic_label'].to_numpy(), s_evt.input_event['instance_label'].astype(int).to_numpy()))
    rets[bw] = pq.compute()
    print(f'(bw = {bw}) pq = {rets[bw]}')

# %%
from tqdm.auto import tqdm
bws = [1.4]
#bws = [0.2]
rets = {}
for bw in bws:
    print(f'bw = {bw}')
    clusterer = MeanShift(use_gpu=False, bandwidth=bw, use_semantic=True, ignore_semantic_labels=(0,))
    pq = PanopticQuality(num_classes=2, ignore_index=None, ignore_semantic_labels=(0,))
    for s_evt, i_evt in tqdm(zip(sem_evts, inst_evts)):
        s_evt.pred_instance_labels = clusterer.cluster(embedding=i_evt.embedding, semantic_labels=s_evt.pred_semantic_labels)
        s_evt.pred_instance_labels = fix_instance_labels(s_evt.pred_instance_labels, s_evt.pred_semantic_labels, labels_to_fix=np.array([0]))
        pq.add((s_evt.pred_semantic_labels, s_evt.pred_instance_labels), (s_evt.input_event['semantic_label'].to_numpy(), s_evt.input_event['instance_label'].astype(int).to_numpy()))
    rets[bw] = pq.compute()
    print(f'(bw = {bw}) pq = {rets[bw]}')
# %%
import plotly
import plotly.express as px
n = 2
evt = sem_evts[n]
i_evt = inst_evts[n]
plot_df = evt.input_event[evt.input_event['semantic_label'] == 1]
plot_df['s_instance_label'] = plot_df['instance_label'].astype(str)
fig = px.scatter_3d(plot_df, x='x', y='y', z='z', color='s_instance_label')
fig.update_traces(marker=dict(size=3),
                  selector=dict(mode='markers'))
fig.show()
# %%
n = 3
evt = inst_evts[n]
evt.pred_semantic_labels = sem_evts[n].pred_semantic_labels
mask = evt.pred_semantic_labels == 1
plot_df = evt.input_event
plot_df['xp'] = evt.embedding[:, 0]
plot_df['yp'] = evt.embedding[:, 1]
plot_df['zp'] = evt.embedding[:, 2]
plot_df['pred_instance_labels'] = sem_evts[n].pred_instance_labels
plot_df = plot_df[mask]
plot_df['s_instance_label'] = plot_df['instance_label'].astype(str)
plot_df['s_pred_instance_labels'] = plot_df['pred_instance_labels'].astype(str)
fig = px.scatter_3d(plot_df, x='xp', y='yp', z='zp', color='s_pred_instance_labels')
fig.update_traces(marker=dict(size=3),
                  selector=dict(mode='markers'))
fig.show()
# %%
fig = px.scatter_3d(plot_df, x='xp', y='yp', z='zp', color='s_instance_label')
fig.update_traces(marker=dict(size=3),
                  selector=dict(mode='markers'))
fig.show()
# %%
plot_df
# %%
plot_df[plot_df['instance_label'] == 4]
# %%
