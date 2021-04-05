from re import L
from hgcal_dev.evaluation.experiments.base_experiment import BaseExperiment, BaseEvent
from hgcal_dev.clustering.meanshift import MeanShift

class HCalEvent(BaseEvent):
    def __init__(self, input_path, pred_path=None, task='panoptic'):
        super().__init__(input_path, pred_path=pred_path, instance_label='cluster', task=task, clusterer=MeanShift(bandwidth=0.01))

class HCalExperiment(BaseExperiment):
    def __init__(self, wandb_version, ckpt_name=None):
        super().__init__(wandb_version, ckpt_name=ckpt_name)

    def make_event(self, input_path, pred_path, task):
        return HCalEvent(input_path, pred_path, task)