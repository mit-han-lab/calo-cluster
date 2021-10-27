
from dataclasses import dataclass
from typing import List
import pandas as pd
import uproot
from calo_cluster.datasets.mixins.calo import CaloDataModule
from tqdm.auto import tqdm


@dataclass
class HCalTTPU200PFDataModuleMixin(CaloDataModule):
    @staticmethod
    def root_to_pickle(root_data_path, raw_data_dir, noise_id=-1):
        ni = 0
        for f in tqdm(sorted(root_data_path.glob('*_pu.root'))):
            root_dir = uproot.open(f)
            root_events = root_dir.get('Events;1')
            df = pd.DataFrame()
            for k, v in root_events[b'RecHit'].items():
                df[k.decode('ascii').split('.')[1]] = v.array()

            for n in range(df.shape[0]):
                jagged_event = df.loc[n]
                df_dict = {k: jagged_event[k] for k in jagged_event.keys()}
                flat_event = pd.DataFrame(df_dict)
                pf_noise_mask = (flat_event['PFcluster0Id'] == noise_id)
                flat_event['pf_hit'] = 0
                flat_event.loc[~pf_noise_mask, 'pf_hit'] = 1
                flat_event.to_pickle(raw_data_dir / f'event_{n+ni:05}.pkl')
            ni += n