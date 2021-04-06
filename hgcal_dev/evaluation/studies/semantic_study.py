from hgcal_dev.evaluation.studies.base_study import BaseStudy
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SemanticStudy(BaseStudy):

    def __init__(self, experiment) -> None:
        super().__init__(experiment)
    
    def confusion_matrix(self, nevents, labels=('hit', 'noise')):
        events = self.experiment.get_events(split='val', n=nevents)
        y_true = np.concatenate([e.input_event[e.class_label] for e in events])
        y_pred = np.concatenate([e.pred_class_labels for e in events])
        data = confusion_matrix(y_true, y_pred)
        fig = sn.heatmap(pd.DataFrame(data, labels, labels)).get_figure()
        fig.savefig(self.out_dir / 'confusion_matrix.png')
        plt.clf()
        norm_data = confusion_matrix(y_true, y_pred, normalize='true')
        fig = sn.heatmap(pd.DataFrame(norm_data, labels, labels)).get_figure()
        fig.savefig(self.out_dir / 'norm_confusion_matrix.png')
        return data