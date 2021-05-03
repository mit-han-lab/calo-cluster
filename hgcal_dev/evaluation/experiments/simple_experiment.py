from hgcal_dev.evaluation.experiments.base_experiment import BaseExperiment, BaseEvent
from hgcal_dev.clustering.meanshift import MeanShift

class SimpleEvent(BaseEvent):
    def __init__(self, input_path, pred_path=None):
        super().__init__(input_path, pred_path=pred_path, instance_label='cluster', task='instance', clusterer=MeanShift(bandwidth=0.01), weight_name='energy')

class SimpleExperiment(BaseExperiment):
    def __init__(self, wandb_version, ckpt_name=None):
        super().__init__(wandb_version, ckpt_name=ckpt_name)

    def make_event(self, input_path, pred_path):
        return SimpleEvent(input_path, pred_path)



def main():
    import uproot
    from tqdm import tqdm
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import plotly
    import plotly.express as px
    import multiprocessing as mp
    from functools import partial
    from hgcal_dev.evaluation.studies.base_study import BaseStudy
    from hgcal_dev.clustering.meanshift import MeanShift
    from sklearn.manifold import TSNE

    exp = SimpleExperiment('2p1ewhmz', 'epoch=14-step=59999.ckpt')

if __name__ == "__main__":
     main()