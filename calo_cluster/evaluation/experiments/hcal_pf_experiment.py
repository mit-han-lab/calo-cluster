from re import L
from calo_cluster.evaluation.experiments.base_experiment import BaseExperiment, BaseEvent
from calo_cluster.clustering.meanshift import MeanShift

class HCalPFEvent(BaseEvent):
    def __init__(self, input_path, instance_label, pred_path=None, task='panoptic'):
        super().__init__(input_path, pred_path=pred_path, semantic_label='hit', instance_label=instance_label, task=task, clusterer=MeanShift(bandwidth=0.01), weight_name='energy')

class HCalPFExperiment(BaseExperiment):
    def __init__(self, wandb_version, ckpt_name=None):
        super().__init__(wandb_version, ckpt_name=ckpt_name)
        if self.multiple_models:
            instance_label = self.cfg[1].dataset.instance_label
        else:
            instance_label = self.cfg.dataset.instance_label
        if instance_label == 'truth':
            self.instance_label = 'trackId'
        elif instance_label == 'antikt':
            self.instance_label = 'RHAntiKtCluster_reco'
        elif instance_label == 'pf':
            self.instance_label = 'PFcluster0Id'
        else:
            raise RuntimeError()

    def make_event(self, input_path, pred_path):
        return HCalPFEvent(input_path=input_path, pred_path=pred_path, task=self.task, instance_label=self.instance_label)