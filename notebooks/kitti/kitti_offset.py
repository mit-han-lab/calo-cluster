# %%
from calo_cluster.clustering.mean_shift_cosine_gpu import L
from calo_cluster.datasets.semantic_kitti import SemanticKITTIOffsetDataModule
from calo_cluster.evaluation.experiments.kitti_offset_experiment import KITTIOffsetExperiment
from calo_cluster.evaluation.metrics.classification import mIoU
from calo_cluster.evaluation.metrics.instance import PanopticQuality
from calo_cluster.clustering.meanshift import MeanShift
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import plotly.express as px
# %%
sem_wandb_version = '19v8uitz'
inst_wandb_version = '36sid3qc'
sem_exp = KITTIOffsetExperiment(sem_wandb_version)
inst_exp = KITTIOffsetExperiment(inst_wandb_version)
# %%
inst_exp.save_predictions(batch_size=16, num_workers=32)
# %%
sem_evts = sem_exp.get_events('val', n=100)
inst_evts = inst_exp.get_events('val', n=100)

# %%
import numpy as np
def num_pred_instances(pred_instance_labels, pred_semantic_labels):
    ret = {}
    unique_semantic_labels = np.unique(pred_semantic_labels)
    for s in unique_semantic_labels:
        mask = pred_semantic_labels == s
        i = pred_instance_labels[mask]
        ret[s] = np.unique(i).shape[0]
    return ret

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
def make_plot(sq, rq, iou, mpq, title):
    pq = sq * rq
    names = ['car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign']
    pq, sq, rq, iou, names = map(list, zip(*sorted(zip(pq, sq, rq, iou, names), reverse=True)))
    score = np.concatenate((iou, rq, sq, pq))
    score[score < 0] = 0
    df = pd.DataFrame({'score': score, 'class': 4*names, 'type': (['iou']*len(names)) + (['rq']*len(names)) + (['sq']*len(names)) + (['pq']*len(names))})
    fig = px.bar(df, x='class', color='type', y='score', barmode='group')
    #fig.add_shape(type="line",
    #xref="paper", yref="paper",
    #x0=0, y0=mpq,
    #x1=1, y1=mpq,
    #line=dict(
    #    color="black",
    #    width=1,
    #    dash='dash'
    #),
    #         )
    #fig.add_annotation(x=18, y=mpq, text='mean PQ', showarrow=True)
    fig.update_layout(title={'text': title,
                            'y':0.95,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'})
    return fig

# %%
iou = mIoU(sem_evts, num_classes=20, ignore_index=19, semantic_label='label_id', reduction='none')
print(iou.mean())

# %%
from tqdm.auto import tqdm
bws = [0.2, 1.2, 1.7, 3.2]
#bws = [0.2]
rets = {}
for bw in bws:
    print(f'bw = {bw}')
    clusterer = MeanShift(bandwidth=bw, use_semantic=True, ignore_semantic_labels=(255,8,9,10,11,12,13,14,15,16,17,18,19))
    pq = PanopticQuality(num_classes=20, ignore_index=19, ignore_semantic_labels=(19,))
    for s_evt, i_evt in tqdm(zip(sem_evts, inst_evts)):
        s_evt.pred_instance_labels = clusterer.cluster(embedding=i_evt.embedding, semantic_labels=s_evt.pred_semantic_labels)
        s_evt.pred_instance_labels = fix_instance_labels(s_evt.pred_instance_labels, s_evt.pred_semantic_labels, labels_to_fix=np.array([8,9,10,11,12,13,14,15,16,17,18]))
        mask = s_evt.input_event['label_id'] == 255
        s_evt.input_event.loc[mask, 'label_id'] = 19
        pq.add((s_evt.pred_semantic_labels, s_evt.pred_instance_labels), (s_evt.input_event['label_id'].to_numpy(), s_evt.input_event['instance_id'].astype(int).to_numpy()))
    rets[bw] = pq.compute()
    print(f'(bw = {bw}) pq = {rets[bw]}')
# %%
ret = rets[0.2]
ret['iou'] = iou
fig = make_plot(ret['sq'][:-1], ret['rq'][:-1], ret['iou'], ret['pq'], title='pred semantic labels, pred instance labels (w/ offset)')
fig.show()
# %%
# pred semantic labels, true instance labels
from tqdm.auto import tqdm
pq = PanopticQuality(num_classes=20, ignore_index=19, ignore_semantic_labels=(19,))
for s_evt, i_evt in tqdm(zip(sem_evts, inst_evts)):
    mask = s_evt.input_event['label_id'] == 255
    s_evt.input_event.loc[mask, 'label_id'] = 19
    pq.add((s_evt.pred_semantic_labels, s_evt.input_event['instance_id'].astype(int).to_numpy()), (s_evt.input_event['label_id'].to_numpy(), s_evt.input_event['instance_id'].astype(int).to_numpy()))
ret = pq.compute()
print(ret)

# %%
ret['iou'] = iou
fig = make_plot(ret['sq'][:-1], ret['rq'][:-1], ret['iou'], ret['pq'], title='pred semantic labels, true instance labels')
fig.show()

# %%
# pred semantic labels, true instance labels (w/ fix)
from tqdm.auto import tqdm
pq = PanopticQuality(num_classes=20, ignore_index=19, ignore_semantic_labels=(19,))
for s_evt, i_evt in tqdm(zip(sem_evts, inst_evts)):
    pred_instance_labels = fix_instance_labels(s_evt.input_event['instance_id'].astype(int).to_numpy(), s_evt.pred_semantic_labels, labels_to_fix=np.array([8,9,10,11,12,13,14,15,16,17,18]))
    mask = s_evt.input_event['label_id'] == 255
    s_evt.input_event.loc[mask, 'label_id'] = 19
    pq.add((s_evt.pred_semantic_labels, pred_instance_labels), (s_evt.input_event['label_id'].to_numpy(), s_evt.input_event['instance_id'].astype(int).to_numpy()))
ret = pq.compute()
print(ret)

# %%
ret['iou'] = iou
fig = make_plot(ret['sq'][:-1], ret['rq'][:-1], ret['iou'], ret['pq'], title='pred semantic labels, true instance labels (w/ fix)')
fig.show()

# %%
# true semantic labels, pred instance labels from coordinates
bws = [0.2, 1.2, 1.7, 3.2]
#bws = [0.2]
rets = {}
for bw in bws:
    print(f'bw = {bw}')
    clusterer = MeanShift(bandwidth=bw, use_semantic=True, ignore_semantic_labels=(255,8,9,10,11,12,13,14,15,16,17,18,19))
    pq = PanopticQuality(num_classes=20, ignore_index=19, ignore_semantic_labels=(19,))
    for s_evt, i_evt in tqdm(zip(sem_evts, inst_evts)):
        s_evt.pred_instance_labels = clusterer.cluster(embedding=i_evt.input_event[['x', 'y', 'z']].to_numpy(), semantic_labels=s_evt.input_event['label_id'].to_numpy())
        s_evt.pred_instance_labels = fix_instance_labels(s_evt.pred_instance_labels, s_evt.input_event['label_id'].to_numpy(), labels_to_fix=np.array([8,9,10,11,12,13,14,15,16,17,18]))
        mask = s_evt.input_event['label_id'] == 255
        s_evt.input_event.loc[mask, 'label_id'] = 19
        pq.add((s_evt.input_event['label_id'].to_numpy(), s_evt.pred_instance_labels), (s_evt.input_event['label_id'].to_numpy(), s_evt.input_event['instance_id'].astype(int).to_numpy()))
    rets[bw] = pq.compute()
    print(f'(bw = {bw}) pq = {rets[bw]}')
# %%
ret = rets[0.2]
ret['iou'] = np.ones(ret['rq'][:-1].shape[0])
fig = make_plot(ret['sq'][:-1], ret['rq'][:-1], ret['iou'], ret['pq'], title='true semantic labels, pred instance labels (no offset)')
fig.show()
# %% 
# true semantic labels, pred instance labels
bws = [0.2, 1.2, 1.7, 3.2]
#bws = [0.2]
rets = {}
for bw in bws:
    print(f'bw = {bw}')
    clusterer = MeanShift(bandwidth=bw, use_semantic=True, ignore_semantic_labels=(255,8,9,10,11,12,13,14,15,16,17,18,19))
    pq = PanopticQuality(num_classes=20, ignore_index=19, ignore_semantic_labels=(19,))
    counts = []
    true_counts = []
    for s_evt, i_evt in tqdm(zip(sem_evts, inst_evts)):
        s_evt.pred_instance_labels = clusterer.cluster(embedding=i_evt.embedding, semantic_labels=s_evt.input_event['label_id'].to_numpy())
        s_evt.pred_instance_labels = fix_instance_labels(s_evt.pred_instance_labels, s_evt.input_event['label_id'].to_numpy(), labels_to_fix=np.array([8,9,10,11,12,13,14,15,16,17,18]))
        mask = s_evt.input_event['label_id'] == 255
        s_evt.input_event.loc[mask, 'label_id'] = 19
        pq.add((s_evt.input_event['label_id'].to_numpy(), s_evt.pred_instance_labels), (s_evt.input_event['label_id'].to_numpy(), s_evt.input_event['instance_id'].astype(int).to_numpy()))
    rets[bw] = pq.compute()
    print(f'(bw = {bw}) pq = {rets[bw]}')

# %%
ret = rets[0.2]
ret['iou'] = np.ones(ret['rq'][:-1].shape[0])
fig = make_plot(ret['sq'][:-1], ret['rq'][:-1], ret['iou'], ret['pq'], title='true semantic labels, pred instance labels (w/ offset)')
fig.show()

# %%
sem_evts[0].input_event
# %%
label_name_mapping = {
    0: 'unlabeled',
    1: 'outlier',
    10: 'car',
    11: 'bicycle',
    13: 'bus',
    15: 'motorcycle',
    16: 'on-rails',
    18: 'truck',
    20: 'other-vehicle',
    30: 'person',
    31: 'bicyclist',
    32: 'motorcyclist',
    40: 'road',
    44: 'parking',
    48: 'sidewalk',
    49: 'other-ground',
    50: 'building',
    51: 'fence',
    52: 'other-structure',
    60: 'lane-marking',
    70: 'vegetation',
    71: 'trunk',
    72: 'terrain',
    80: 'pole',
    81: 'traffic-sign',
    99: 'other-object',
    252: 'moving-car',
    253: 'moving-bicyclist',
    254: 'moving-person',
    255: 'moving-motorcyclist',
    256: 'moving-on-rails',
    257: 'moving-bus',
    258: 'moving-truck',
    259: 'moving-other-vehicle'
}
# %%
label_names = {sem_exp.datamodule.label_map[k]: v for k,v in label_name_mapping.items()}
# %%
names = ['car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign', 'other']
# %%
things_ids = set([10, 11, 15, 18, 20, 30, 31, 32, 252, 253, 254, 255, 258, 259])
things_names = [label_name_mapping[i] for i in things_ids]
# %%
things_names

# %%

# %%
# check true instance labels for stuff
stuff_names = ['road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign']
id_to_name = {i+8: v for i,v in enumerate(stuff_names)}
for evt in sem_evts:
    instance_labels = evt.input_event['instance_id'].astype(int).to_numpy()
    semantic_labels = evt.input_event['label_id'].to_numpy()
    unique_labels = np.unique(semantic_labels)
    counts = {}
    for l in unique_labels:
        if l not in id_to_name.keys():
            continue
        mask = semantic_labels == l
        counts[l] = np.unique(instance_labels[mask]).shape[0]
    for k,v in counts.items():
        if v != 1:
            print(k)
print('done!')
# %%
# check true instance labels for stuff
stuff_names = ['road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign']
id_to_name = {i+8: v for i,v in enumerate(stuff_names)}
for evt in sem_evts:
    instance_labels = evt.input_event['instance_id'].astype(int).to_numpy()
    semantic_labels = evt.input_event['label_id'].to_numpy()
    unique_labels = np.unique(semantic_labels)
    counts = {}
    for l in unique_labels:
        if l not in id_to_name.keys():
            continue
        mask = semantic_labels == l
        i_labels = np.unique(instance_labels[mask])
        bad_labels = i_labels[i_labels < 0]
        if bad_labels.shape[0] != 0:
            print(f'{id_to_name[l]}: {bad_labels}')
print('done!')
# %%
pred_offsets = inst_evts[0].embedding - inst_evts[0].input_event[['x', 'y', 'z']]
# %%
pred_offsets.abs().mean(axis=0)
# %%

# %%
dm = SemanticKITTIOffsetDataModule.from_config()
# %%
dm.setup('fit')
# %%
