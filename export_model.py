from hgcal_dev.evaluation.studies.clustering_study import ClusteringStudy
from hgcal_dev.evaluation.experiments.antikt_hcal_experiment import AntiKtHCalExperiment
from hgcal_dev.evaluation.experiments.hcal_pf_experiment import HCalPFExperiment
from hgcal_dev.evaluation.experiments.pf_hcal_pf_experiment import PFHCalPFExperiment
from hgcal_dev.evaluation.studies.panoptic_study import PanopticStudy
from hgcal_dev.clustering.meanshift import MeanShift
from hgcal_dev.clustering.identity_clusterer import IdentityClusterer
from hgcal_dev.datasets.hcal_pf import HCalPFDataModule
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sn
from hgcal_dev.evaluation.utils import get_palette
from hgcal_dev.evaluation.studies import functional as F


def main():
    exp = HCalPFExperiment(wandb_version='3bybg4l1')
    del exp.model.embed_criterion
    dataloader = exp.datamodule.val_dataloader()
    feats = dataloader.dataset[0]['features']
    onnx_path = exp.ckpt_path.parent.parent / 'model.onnx'
    exp.model.to_onnx(file_path=onnx_path, input_sample=feats, export_params=True)
    #script = exp.model.to_torchscript(example_inputs=feats)

if __name__ == "__main__":
    main()