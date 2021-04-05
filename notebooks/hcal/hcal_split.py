# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
sys.path.append('/home/alexj/hgcal-dev')


# %%
from hgcal_dev.visualization.hcal import HCalEvent
import numpy as np
import pandas as pd
import torch
from hgcal_dev.utils.experiment import Experiment
from pathlib import Path
from hgcal_dev.clustering.meanshift import MeanShift
from tqdm import tqdm
from sklearn.cluster import estimate_bandwidth
from sklearn.manifold import TSNE
import plotly
import plotly.express as px
import colorsys
from hgcal_dev.training.criterion import centroid_instance_loss
from sklearn import cluster


# %%
run_path = Path('/home/alexj/outputs/wandb/run-20210216_033726-3r6mg438')
ckpt_path = Path('/home/alexj/outputs/wandb/run-20210216_033726-3r6mg438/files/hgcal-spvcnn/3r6mg438/checkpoints/epoch=4-v0.ckpt')
experiment = Experiment(run_path, ckpt_path)


# %%
def top_n_clusters(events: list, n: int):
    out = []
    for event in events:
        out.append(event.top_n_clusters(n))
    return out


# %%
def evaluate_pqs(events, max_clusters, bandwidth):
    n_clusters = list(range(1, max_clusters+1))
    pqs = np.zeros(len(n_clusters))
    for i, n in enumerate(tqdm(n_clusters)):
        filtered_events = top_n_clusters(events, n)
        clusterer = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
        for j, event in enumerate(filtered_events):
            pred_instance_labels = clusterer.fit_predict(event.embedding)
            sq, rq, _ = event.pq(pred_instance_labels)
            mask = (sq != 0.0) & (rq != 0.0)
            sq, rq = sq[mask], rq[mask]
            pq = (sq * rq).mean() / len(filtered_events)
            if np.isnan(pq):
                print(j)
                breakpoint()
            pqs[i] += (sq * rq).mean() / len(filtered_events)
    return pqs, n_clusters


# %%
events = experiment.get_events(split='val', n=40)
bandwidth = 0.38
n = 3
filtered_events = top_n_clusters(events, n)
event = filtered_events[27]
clusterer = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
pred_instance_labels = clusterer.fit_predict(event.embedding)

# %%
sq, rq, _ = event.pq(pred_instance_labels)
mask = (sq != 0.0) & (rq != 0.0)
sq, rq = sq[mask], rq[mask]
(sq * rq).mean()


# %%
events = experiment.get_events(split='val', n=40)
evaluate_pqs(events, 4, 0.38)


# %%
event.plot_3d(semantic=False, scale_size=False, truth=True, pred_instance_labels=pred_instance_labels)


# %%
event.plot_3d(semantic=False, scale_size=False, truth=False, pred_instance_labels=pred_instance_labels)


# %%
event.plot_3d(semantic=True, scale_size=False, truth=True, pred_instance_labels=pred_instance_labels)


# %%
event.plot_3d(semantic=True, scale_size=False, truth=True, pred_instance_labels=pred_instance_labels)


# %%
n_clusters = list(range(1, max_clusters+1))
pqs = np.zeros(len(n_clusters))


# %%
def pq_study(experiment, max_clusters, n_events, bandwidth):
    events = experiment.get_events(split='val', n=n_events)
    pqs, n_clusters = evaluate_pqs(events, max_clusters, bandwidth)

    pq_fig = px.scatter(x=n_clusters, y=pqs, labels={'x': 'n_clusters', 'y': 'pq'}, title='PQ vs. Top k Clusters')

    return pq_fig


# %%
pq_study(experiment, 2, 1, 0.225)


# %%
events = experiment.get_events(split='val', n=1)


# %%
filtered_events = top_n_clusters(events, 2)


# %%
events = experiment.get_events('train', n=10)


# %%
event = events[0]
train_pred_tsne = TSNE(n_jobs=-1).fit_transform(event.embedding)


# %%
fig = event.plot_3d(semantic=False, scale_size=False)
fig.update_traces(mode='markers', marker_size=3)
fig.show()


# %%
clusterer = MeanShift(use_gpu=False, bandwidth=0.36)
pred_instance_labels = clusterer.cluster(event.embedding)
fig = event.plot_3d(semantic=False, scale_size=False, truth=False, pred_instance_labels=pred_instance_labels)
fig.update_traces(mode='markers', marker_size=3)
fig.show()


# %%
palette = 'rgb(120,62,62), rgb(94,79,49), rgb(53,69,53), rgb(56,90,107), rgb(80,56,107), rgb(82,42,42), rgb(69,64,53), rgb(92,120,98), rgb(82,97,107), rgb(63,42,82), rgb(120,92,92), rgb(56,49,29), rgb(62,120,81), rgb(63,74,82), rgb(91,73,94), rgb(69,53,53), rgb(94,89,73), rgb(49,94,67), rgb(56,83,107), rgb(120,62,120), rgb(94,55,49), rgb(120,114,92), rgb(73,94,81), rgb(42,63,82), rgb(69,36,69), rgb(56,33,29), rgb(107,100,56), rgb(36,69,51), rgb(29,44,56), rgb(94,49,91), rgb(94,76,73), rgb(82,76,42), rgb(43,56,50), rgb(49,67,94), rgb(56,29,54), rgb(120,78,62), rgb(120,120,62), rgb(92,120,109), rgb(36,49,69), rgb(120,92,116), rgb(69,45,36), rgb(68,69,53), rgb(56,107,90), rgb(92,103,120), rgb(56,43,54), rgb(120,100,92), rgb(91,94,73), rgb(29,56,49), rgb(62,81,120), rgb(82,63,75), rgb(94,64,49), rgb(85,94,49), rgb(62,120,112), rgb(49,55,94), rgb(120,62,97), rgb(94,80,73), rgb(60,69,36), rgb(49,94,88), rgb(29,33,56), rgb(82,42,63), rgb(120,85,62), rgb(112,120,92), rgb(53,69,67), rgb(92,96,120), rgb(56,29,42), rgb(56,40,29), rgb(45,56,29), rgb(36,69,67), rgb(73,76,94), rgb(107,56,73), rgb(94,70,49), rgb(51,56,43), rgb(73,94,93), rgb(53,54,69), rgb(69,36,45), rgb(120,105,92), rgb(81,120,62), rgb(56,104,107), rgb(36,36,69), rgb(120,92,100), rgb(56,49,43), rgb(45,69,36), rgb(92,118,120), rgb(59,56,107), rgb(94,73,78), rgb(120,93,62), rgb(78,94,73), rgb(43,54,56), rgb(91,82,107), rgb(94,49,55), rgb(69,53,36), rgb(55,94,49), rgb(49,85,94), rgb(40,29,56), rgb(56,29,31), rgb(120,101,62), rgb(29,56,29), rgb(36,62,69), rgb(70,63,82)'.split(', ')


# %%
fig = px.scatter(x=train_pred_tsne[:, 0], y=train_pred_tsne[:, 1], color=event.input_event['labels_i'].astype('str'), color_discrete_sequence=palette, size=event.input_event['energy'])
fig.show()


# %%
losses = np.zeros(len(events))
for i, event in enumerate(events):
    losses[i] = centroid_instance_loss(torch.tensor(event.embedding), torch.tensor(event.input_event['labels_i']))[0]


# %%
losses


# %%
HCalEvent.plot_pqs_vs_bandwidth(events, bws=[0.001, 0.01, 0.1, 1.0], log_x=True)


# %%
HCalEvent.plot_pqs_vs_bandwidth(events, bws=[0.05, 0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])


# %%
HCalEvent.plot_pqs_vs_bandwidth(events, bws=[0.3, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46])


# %%
event.embedding.shape


# %%
def tsne_pred(bw):
    clusterer = MeanShift(use_gpu=False, bandwidth=bw)
    pred_instance_labels = clusterer.cluster(event.embedding)
    fig = px.scatter(x=train_pred_tsne[:, 0], y=train_pred_tsne[:, 1], color=pred_instance_labels.astype('str'), color_discrete_sequence=palette, size=event.input_event['energy'])
    fig.show()


# %%
fig = tsne_pred(0.36)


# %%
experiment.save('hcal_tsne_pred.png')


# %%
HCalEvent.plot_confusion_matrix(events)


# %%



