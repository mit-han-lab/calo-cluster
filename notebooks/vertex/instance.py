# %%
from calo_cluster.evaluation.experiments.base_experiment import BaseExperiment
from pathlib import Path 
from calo_cluster.clustering.meanshift import MeanShift
from tqdm.auto import tqdm
from calo_cluster.evaluation.metrics.instance import PanopticQuality
# %%
# Load the trained model -- replace wandb_version as appropriate.
wandb_version = '2s303bl3'
exp = BaseExperiment(wandb_version)

# %%
# Load the validation events.
evts = exp.get_events('val')

#%%
print("Test CPU version of MeanShift")
# bds = [0.001, 0.01, 0.1, 1, 10, 100]
# bds = [2, 4, 6, 8, 10, 15, 20, 30, 50]
bds = [6+ 0.1*i for i in range(11)]

for bd in bds:
    clusterer = MeanShift(bandwidth=bd, use_gpu=False, use_semantic=False)
    for evt in tqdm(evts[0:200]):
        evt.pred_instance_labels = clusterer.cluster(embedding=evt.embedding)

    pq = PanopticQuality(num_classes=2, semantic=False)
    for evt in tqdm(evts[0:200]):
        pq.add(evt.pred_instance_labels, evt.input_event['truth_vtxID'].astype(int).to_numpy())
    print(f'bd = {bd}, pq = {pq.compute()}')

# bd between 6 and 8 is good for voxel size 0.05 and 5 coordinates 
#%%
# %%
print("Using GPU version of MeanShift")
# Run the mean shift clustering to get instance labels.
clusterer = MeanShift(bandwidth=0.05, use_semantic=False)
for evt in tqdm(evts):
    evt.pred_instance_labels = clusterer.cluster(embedding=evt.embedding)
    
# %%
# Calculate the panoptic quality, ignoring the noise.
pq = PanopticQuality(num_classes=2, semantic=False)
for evt in tqdm(evts):
    pq.add(evt.pred_instance_labels, evt.input_event['truth_vtxID'].astype(int).to_numpy())
print(f'pq = {pq.compute()}')

# %%
# Plot the predicted and true data.
plot_df = evts[0].input_event
plot_df['pred_label'] = evts[0].pred_instance_labels
plot_df['pred_label'] = plot_df['pred_label'].astype(str)
plot_df['instance_label'] = plot_df['truth_vtxID'].astype(str)
import plotly.express as px
fig_pred = px.scatter_3d(plot_df, x='d0', y='z0', z='qp', color='pred_label')
# %%
fig_truth = px.scatter_3d(plot_df, x='d0', y='z0', z='qp', color='instance_label')


# %%
fig_pred.write_image("images/fig_pred_gpu.png")
fig_truth.write_image("images/fig_truth_gpu.png")


# %%
clusterer = MeanShift(bandwidth=6.5, use_gpu=False, use_semantic=False)
for evt in tqdm(evts):
    evt.pred_instance_labels = clusterer.cluster(embedding=evt.embedding)

pq = PanopticQuality(num_classes=2, semantic=False)
for evt in tqdm(evts):
    pq.add(evt.pred_instance_labels, evt.input_event['truth_vtxID'].astype(int).to_numpy())
print(f'bd = 6.5, pq = {pq.compute()}')

# %%
plot_df = evts[0].input_event
plot_df['pred_label'] = evts[0].pred_instance_labels
plot_df['pred_label'] = plot_df['pred_label'].astype(str)
plot_df['instance_label'] = plot_df['truth_vtxID'].astype(str)
import plotly.express as px
fig_pred = px.scatter_3d(plot_df, x='d0', y='z0', z='qp', color='pred_label')
fig_truth = px.scatter_3d(plot_df, x='d0', y='z0', z='qp', color='instance_label')

# %%
fig_pred.write_image("images/fig_pred_cpu_6p5.png")
fig_truth.write_image("images/fig_truth_cpu_6p5.png")

# %%
print("test the AMVF PQ score")

pq_amvf = PanopticQuality(num_classes=2, semantic=True, ignore_semantic_labels=[0])
for evt in tqdm(evts):
    pq_amvf.add((evt.input_event['reco_semantic_label'].astype(int).to_numpy(), evt.input_event['reco_AMVF_vtxID'].astype(int).to_numpy()),
    (evt.input_event['truth_semantic_label'].astype(int).to_numpy(), evt.input_event['truth_vtxID'].astype(int).to_numpy()))
print(f'for amvf, pq = {pq_amvf.compute()}')

# %%
# Draw AMVF plots 
plot_df = evts[0].input_event
plot_df['reco_AMVF_vtxID'] = plot_df['reco_AMVF_vtxID'].astype(str)
import plotly.express as px
fig_amvf = px.scatter_3d(plot_df, x='d0', y='z0', z='qp', color='reco_AMVF_vtxID')
fig_amvf.write_image("images/fig_amvf.png")

# %%
import ROOT 
import numpy as np
# %%
ROOT.gROOT.SetStyle("ATLAS")
canvas = ROOT.TCanvas("c1","Profile histogram",200,10,700,500)

hprof_amvf = ROOT.TProfile("hprof_amvf","Profile of N amvf vtx versus N truth",21,-0.5,20.5,0,20)
hprof_spvcnn = ROOT.TProfile("hprof_spvcnn","Profile of N spvcnn vtx versus N truth",21,-0.5,20.5,0,20)

for evt in tqdm(evts):
    hprof_amvf.Fill(np.max(evt.input_event['truth_vtxID'])+1, np.max(evt.input_event['reco_AMVF_vtxID'])+1)
    hprof_spvcnn.Fill(np.max(evt.input_event['truth_vtxID'])+1, np.max(evt.pred_instance_labels)+1)

hprof_amvf.GetXaxis().SetTitle("Number of #it{pp} interactions per bunch crossing")
hprof_amvf.GetYaxis().SetTitle("Average number of reconstructed veftices")

hprof_amvf.SetTitle("Profile of N reco vtx versus N truth")
hprof_amvf.Draw()

hprof_spvcnn.SetLineColor(2)
hprof_spvcnn.SetMarkerColor(2)
hprof_spvcnn.Draw("same")

legend = ROOT.TLegend(0.7,0.60,0.9,0.70)

legend.AddEntry(hprof_amvf,"AMVF","lp")
legend.AddEntry(hprof_spvcnn,"SPVCNN","lp")
legend.Draw("same")

ll=ROOT.TLine(0.,0.,15.,15.);
ll.SetLineWidth(2);
ll.SetLineColor(13);
ll.SetLineStyle(9);
ll.Draw("same");

text = ROOT.TText(10, 10, "100\% interaction reconstruction efficiency");

text.SetTextColor(13);
text.SetTextAngle(40);
text.SetTextSize(0.04);
text.Draw("same");

canvas.Draw()
# %%
np.max(evts[0].input_event['reco_AMVF_vtxID'])+1

# %%
np.max(evts[0].input_event['truth_vtxID'])+1

# %%
np.max(evts[0].pred_instance_labels)+1

# %%
type(evts[0])
# %%
type(evts.pred_instance_labels)
# %%
