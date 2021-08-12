from re import L
from calo_cluster.evaluation.experiments.base_experiment import BaseExperiment, BaseEvent
from calo_cluster.clustering.meanshift import MeanShift

class HCalEvent(BaseEvent):
    def __init__(self, input_path, instance_label, pred_path=None, task='panoptic'):
        super().__init__(input_path, pred_path=pred_path, semantic_label='hit', instance_label=instance_label, task=task, clusterer=MeanShift(bandwidth=0.01), weight_name='energy')

class HCalExperiment(BaseExperiment):
    def __init__(self, wandb_version, ckpt_name=None):
        super().__init__(wandb_version, ckpt_name=ckpt_name)
        if self.multiple_models:
            instance_label = self.cfg[1].dataset.instance_label
        else:
            instance_label = self.cfg.dataset.instance_label
        if instance_label == 'truth':
            self.instance_label = 'trackId'
        else:
            self.instance_label = 'RHAntiKtCluster'

    def make_event(self, input_path, pred_path):
        return HCalEvent(input_path=input_path, pred_path=pred_path, task=self.task, instance_label=self.instance_label)