# %%
from calo_cluster.datasets.hcal_Zll_jets import HCalZllJetsDataModule
from pathlib import Path


# %%
root_data_path = Path('/global/cscratch1/sd/schuya/calo_cluster/data/hcal/Zll_jets')
raw_data_dir = root_data_path / 'min_energy_0.0_min_hits_0'


# %%
HCalZllJetsDataModule.root_to_pickle(root_data_path, raw_data_dir)
# %%
