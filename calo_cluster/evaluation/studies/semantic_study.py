from calo_cluster.evaluation.studies.base_study import BaseStudy
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SemanticStudy(BaseStudy):

    def __init__(self, experiment) -> None:
        super().__init__(experiment)
    
    def confusion_matrix(self, nevents=-1, labels=('hit', 'noise')):
        events = self.experiment.get_events(split='val', n=nevents)
        y_true = np.concatenate([e.input_event[e.semantic_label] for e in events])
        y_pred = np.concatenate([e.pred_semantic_labels for e in events])
        F1 = f1_score(y_true, y_pred)
        data = confusion_matrix(y_true, y_pred)
        ax = sn.heatmap(pd.DataFrame(data, labels, labels), annot=True)
        ax.set_title(f'F1 = {F1}')
        ax.set_xlabel('predicted')
        ax.set_ylabel('true')
        fig = ax.get_figure()
        fig.savefig(self.out_dir / 'confusion_matrix.png', facecolor='white', dpi=1200)
        plt.clf()
        norm_data = confusion_matrix(y_true, y_pred, normalize='true')
        ax = sn.heatmap(pd.DataFrame(norm_data, labels, labels), annot=True)
        ax.set_title(f'F1 = {F1}')
        ax.set_xlabel('predicted')
        ax.set_ylabel('true')
        fig = ax.get_figure()
        fig.savefig(self.out_dir / 'norm_confusion_matrix.png', facecolor='white', dpi=1200)
        return data, norm_data