# %%
from calo_cluster.evaluation.experiments.semantic_kitti_experiment import SemanticKITTIExperiment
from calo_cluster.evaluation.metrics.classification import mIoU
from calo_cluster.evaluation.metrics.instance import PanopticQuality
from calo_cluster.clustering.meanshift import MeanShift
# %%
wandb_version = '3ghelecj'
exp = SemanticKITTIExperiment(wandb_version)

# %%
exp.save_predictions(batch_size=32)
# %%
evts = exp.get_events('val', n=-1)

# %%
from tqdm.auto import tqdm
clusterer = MeanShift(bandwidth=0.05, use_semantic=True, ignore_semantic_labels=(255,19))
pq = PanopticQuality(num_classes=20, ignore_index=19, ignore_semantic_labels=(19,))
for evt in tqdm(evts):
    evt.pred_instance_labels = clusterer.cluster(embedding=evt.embedding, semantic_labels=evt.pred_semantic_labels)
    mask = evt.input_event['label_id'] == 255
    evt.input_event.loc[mask, 'label_id'] = 19
    pq.add((evt.pred_semantic_labels, evt.pred_instance_labels), (evt.input_event['label_id'].to_numpy(), evt.input_event['instance_id'].astype(int).to_numpy()))
print(f'pq = {pq.compute()}')
# %%
iou = mIoU(evts, num_classes=20, ignore_index=19, semantic_label='label_id', reduction='none')
print(iou.mean())

# %%
import numpy as np
np.unique(evts[0].pred_semantic_labels)
# %%
import numpy as np
from calo_cluster.evaluation.metrics.instance import PanopticQuality

xs, xi = [], []
ys, yi = [], []
# some ignore_index stuff
N_ignore_index = 50
xs.extend([-1 for i in range(N_ignore_index)])
xi.extend([0 for i in range(N_ignore_index)])
ys.extend([-1 for i in range(N_ignore_index)])
yi.extend([0 for i in range(N_ignore_index)])
# grass segment
N_grass = 50
N_grass_x = 40
xs.extend([0 for i in range(N_grass_x)])
xs.extend([1 for i in range(N_grass - N_grass_x)])
xi.extend([0 for i in range(N_grass)])
ys.extend([0 for i in range(N_grass)])
yi.extend([0 for i in range(N_grass)])
# sky segment
N_sky = 50
N_sky_x = 40
xs.extend([1 for i in range(N_sky_x)])
xs.extend([0 for i in range(N_sky - N_sky_x)])
xi.extend([0 for i in range(N_sky)])
ys.extend([1 for i in range(N_sky)])
yi.extend([0 for i in range(N_sky)])
# wrong dog as person xiction
N_dog = 50
N_person = N_dog
xs.extend([2 for i in range(N_person)])
xi.extend([35 for i in range(N_person)])
ys.extend([3 for i in range(N_dog)])
yi.extend([22 for i in range(N_dog)])
N_person = 50
xs.extend([2 for i in range(6 * N_person)])
xi.extend([8 for i in range(4 * N_person)])
xi.extend([95 for i in range(2 * N_person)])
ys.extend([2 for i in range(6 * N_person)])
yi.extend([33 for i in range(3 * N_person)])
yi.extend([42 for i in range(N_person)])
yi.extend([11 for i in range(2 * N_person)])
xs = np.array(xs, dtype=np.int64).reshape(1, -1)
xi = np.array(xi, dtype=np.int64).reshape(1, -1)
ys = np.array(ys, dtype=np.int64).reshape(1, -1)
yi = np.array(yi, dtype=np.int64).reshape(1, -1)
evaluator = PanopticQuality(num_classes=4, ignore_index=-1)
evaluator.add((xs, xi), (ys, yi))
pq = evaluator.compute()['pq']
# %%
