from hgcal_dev.evaluation.studies.semantic_study import SemanticStudy
from hgcal_dev.evaluation.studies.clustering_study import ClusteringStudy
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PanopticStudy(SemanticStudy, ClusteringStudy):

    def __init__(self, experiment) -> None:
        super().__init__(experiment)