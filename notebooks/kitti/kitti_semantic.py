# %%
from calo_cluster.evaluation.experiments.semantic_kitti_experiment import SemanticKITTIExperiment
from calo_cluster.evaluation.metrics.classification import mIoU
# %%
wandb_version = '1ui4qrxw'
exp = SemanticKITTIExperiment(wandb_version)

# %%
exp.save_predictions(batch_size=16)
# %%
evts = exp.get_events('val')
from tqdm import tqdm
for evt in tqdm(evts):
    mask = evt.input_event['label_id'] == 255
    evt.input_event.loc[mask, 'label_id'] = 19
# %%
iou = mIoU(evts, num_classes=20, ignore_index=19, semantic_label='label_id')
# %%
evts[0].input_event['label_id'].unique()
# %%
evts[0].input_event['label_id'].shape
# %%
evts[0].pred_semantic_labels.shape
# %%
