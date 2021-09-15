# %%
from calo_cluster.evaluation.experiments.semantic_kitti_experiment import SemanticKITTIExperiment
from calo_cluster.evaluation.metrics.classification import mIoU
# %%
wandb_version = '23d0pqcm'
ckpt_name = '13-16743.ckpt'
exp = SemanticKITTIExperiment(wandb_version, ckpt_name=ckpt_name)

# %%
exp.save_predictions(batch_size=16)
# %%
evts = exp.get_events('val')
from tqdm import tqdm
for evt in tqdm(evts):
    mask = evt.input_event['label_id'] == 255
    evt.input_event.loc[mask, 'label_id'] = 19
# %%
iou = mIoU(evts, num_classes=20, ignore_index=19, semantic_label='label_id', reduction='none')
print(iou.mean())
# %%
evts[0].input_event['label_id'].unique()
# %%
evts[0].input_event['label_id'].shape
# %%
evts[0].pred_semantic_labels.shape
# %%
evt2 = exp.get_events('val')
# %%
for i in iou:
    print(i)
# %%
len(iou)
# %%
iou.mean()
# %%
