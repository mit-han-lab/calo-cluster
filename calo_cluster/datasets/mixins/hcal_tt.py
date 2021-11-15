from dataclasses import dataclass

import pandas as pd
import uproot
from tqdm.auto import tqdm

from calo_cluster.datasets.mixins.calo import CaloDataModule
from calo_cluster.datasets.pandas_data import PandasDataset


@dataclass
class HCalTTMixin(PandasDataset):
    instance_target: str

    def __post_init__(self):
        if self.instance_target == 'truth':
            self.semantic_label = 'hit'
            self.instance_label = 'trackId'
        elif self.instance_target == 'antikt':
            raise NotImplementedError()
            #instance_label = 'RHAntiKtCluster_reco'
        elif self.instance_target == 'pf':
            self.semantic_label = 'pf_hit'
            self.instance_label = 'PFcluster0Id'
        else:
            raise RuntimeError()
        return super().__post_init__()


@dataclass
class HCalTTDataModuleMixin(CaloDataModule):
    instance_target: str

    def make_dataset_kwargs(self) -> dict:
        kwargs = {
            'instance_target': self.instance_target
        }
        kwargs.update(super().make_dataset_kwargs())
        return kwargs

    @staticmethod
    def root_to_pickle(root_data_path, raw_data_dir, noise_id=-99, pf_noise_id=-1):
        ni = 0
        for f in sorted(root_data_path.glob('*.root')):
            root_dir = uproot.open(f)
            root_events = root_dir.get('Events;1')
            data_dict = {}
            for k, v in tqdm(root_events['RecHit'].items()):
                data_dict[k.split('.')[1]] = v.array()

            for n in tqdm(range(len(data_dict[list(data_dict.keys())[0]]))):
                df_dict = {k: data_dict[k][n].to_numpy() for k in data_dict.keys() if k != 'fBits'}
                flat_event = pd.DataFrame(df_dict)
                pf_noise_mask = (flat_event['PFcluster0Id'] == pf_noise_id)
                flat_event['pf_hit'] = 0
                flat_event.loc[~pf_noise_mask, 'pf_hit'] = 1
                flat_event.astype({'hit': int})
                hit_mask = (flat_event['genE'] > 0.2)
                noise_mask = ~hit_mask
                flat_event['hit'] = hit_mask.astype(int)
                flat_event.loc[noise_mask, 'trackId'] = noise_id
                flat_event.to_pickle(raw_data_dir / f'event_{n+ni:05}.pkl')
            ni += n + 1
