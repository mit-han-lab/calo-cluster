# %%
from calo_cluster.evaluation.experiments.kitti_offset_experiment import KITTIOffsetExperiment
from calo_cluster.evaluation.metrics.classification import mIoU
from calo_cluster.evaluation.metrics.instance import PanopticQuality
from calo_cluster.clustering.meanshift import MeanShift
# %%
wandb_version = '31p586i3'
exp = KITTIOffsetExperiment(wandb_version)

# %%
exp.save_predictions(batch_size=32, num_workers=64)
# %%
evts = exp.get_events('val', n=1000)

# %%
from tqdm import tqdm
bws = [0.2, 1.2, 1.7, 3.2]
for bw in bws:
    print(f'bw = {bw}')
    clusterer = MeanShift(bandwidth=bw, use_semantic=True, ignore_semantic_labels=(255,19))
    pq = PanopticQuality(num_classes=20, ignore_index=19, ignore_semantic_labels=(19,))
    for evt in tqdm(evts):
        evt.pred_instance_labels = clusterer.cluster(embedding=evt.embedding, semantic_labels=evt.pred_semantic_labels)
        mask = evt.input_event['label_id'] == 255
        evt.input_event.loc[mask, 'label_id'] = 19
        pq.add((evt.pred_semantic_labels, evt.pred_instance_labels), (evt.input_event['label_id'].to_numpy(), evt.input_event['instance_id'].astype(int).to_numpy()))
    print(f'(bw = {bw}) pq = {pq.compute()}')
# %%
iou = mIoU(evts, num_classes=20, ignore_index=19, semantic_label='label_id', reduction='none')
print(iou.mean())

# %%
exp.datamodule.label_map
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
label_names = {exp.datamodule.label_map[k]: v for k,v in label_name_mapping.items()}
# %%
label_names
# %%
import numpy as np
results = {'sq': np.array([0.66796851,  0.64717588,  0.83680809,  0.72256327,  0.7649425 ,0.78918475,  0.8228926 ,  0.        ,  0.94842795,  0.66985639,0.84646327,  0.        ,  0.88814117,  0.69580634,  0.90514014,0.66572518,  0.71240592,  0.70947434,  0.7443778 , -1.        ]), 'rq': np.array([ 0.07377455,  0.00746269,  0.37851662,  0.20087336,  0.21163012, 0.26406685,  0.39837398,  0.        ,  1.        ,  0.10909091, 0.978,  0,  0.925     ,  0.25311203,  1.        , 0.51130085,  0.44701155,  0.731     ,  0.51228978, -1.        ])}
# %%
results
# %%
label_names.values()
# %%
names = ['moving-car', 'bicycle', 'motorcycle', 'moving-truck', 'moving-other-vehicle', 'moving-person', 'moving-bicyclist', 'moving-motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign', 'other']
# %%
for sq, rq, name in zip(results['sq'], results['rq'], names):
    print(f'{name}: {sq} sq, {rq} rq')
# %%
things_ids = set([10, 11, 15, 18, 20, 30, 31, 32, 252, 253, 254, 255, 258, 259])
things_names = [label_name_mapping[i] for i in things_ids]
# %%
things_names
# %%
