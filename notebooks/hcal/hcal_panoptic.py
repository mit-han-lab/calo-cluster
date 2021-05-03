# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from hgcal_dev.evaluation.experiments.hcal_experiment import HCalExperiment
from hgcal_dev.evaluation.studies.panoptic_study import PanopticStudy
from hgcal_dev.clustering.meanshift import MeanShift

# %%
exp = HCalExperiment('1w3dn59r')


# %%
ps = PanopticStudy(exp)


# %%
ps.clusterer = MeanShift(bandwidth=0.022)
ps.pq(nevents=1000, use_weights=True, ignore_class_labels=(0,))


# %%
ps.confusion_matrix(nevents=1000)


# %%
ps.clusterer = MeanShift(bandwidth=0.022)
ps.response(nevents=1000, splits=('val',), nbins=100, lo=0.5, hi=100, match_highest=True)


# %%
exp2 = HCalExperiment('1w3dn59r', 'epoch=2-step=61213.ckpt')


# %%
cs2 = ClusteringStudy(exp2)


# %%
cs2.resolution()


# %%
ret_best = exp.loss('val')


# %%
ret = exp2.loss('val')


# %%
ret_small = exp2.loss('val', batch_size=1)


# %%
import plotly
import plotly.express as px
import pandas as pd


# %%
df = pd.DataFrame(ret)
px.histogram(df, x='class_loss')


# %%
px.histogram(df, x='embed_loss')


# %%
df_small = pd.DataFrame(ret_small)
px.histogram(df_small, x='class_loss')


# %%
px.histogram(df_small, x='embed_loss')


# %%



