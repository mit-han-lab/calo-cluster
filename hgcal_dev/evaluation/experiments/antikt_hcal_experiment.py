from re import L
from hgcal_dev.evaluation.experiments.base_experiment import BaseExperiment, BaseEvent
from hgcal_dev.clustering.meanshift import MeanShift
import pandas as pd

class AntiKtHCalEvent(BaseEvent):
    def __init__(self, input_path, pred_path=None, task='instance'):
        super().__init__(input_path, pred_path=pred_path, class_label='hit', instance_label='trackId', task=task, weight_name='energy')

    def _load(self):
        input_event = pd.read_pickle(self.input_path)
        self.embedding = input_event['RHAntiKtCluster_reco'].values
        self.input_event = input_event

class AntiKtHCalExperiment(BaseExperiment):
    def __init__(self, wandb_version, ckpt_name=None):
        super().__init__(wandb_version, ckpt_name=ckpt_name)
        self.task = 'instance'

    def make_event(self, input_path, pred_path):
        return AntiKtHCalEvent(input_path=input_path, pred_path=pred_path, task='instance')

    def save_predictions(self):
        pass