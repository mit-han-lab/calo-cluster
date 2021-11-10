
from dataclasses import dataclass

import pandas as pd
import uproot
from tqdm.auto import tqdm

from calo_cluster.datasets.mixins.calo import CaloDataModule


@dataclass
class HCalTTPU200PFDataModuleMixin(CaloDataModule):
    @staticmethod
    def root_to_pickle(root_data_path, raw_data_dir, noise_id=-1):
        raw_data_dir.mkdir(exist_ok=True, parents=True)
        ni = 0
        for f in tqdm(sorted(root_data_path.glob('*_pu.root'))):
            root_dir = uproot.open(f)
            root_events = root_dir.get('Events;1')
            data_dict = {}
            for k, v in tqdm(root_events['RecHit'].items()):
                data_dict[k.split('.')[1]] = v.array()

            for n in tqdm(range(len(data_dict[list(data_dict.keys())[0]]))):
                df_dict = {k: data_dict[k][n].to_numpy() for k in data_dict.keys() if k != 'fBits'}
                flat_event = pd.DataFrame(df_dict)
                pf_noise_mask = (flat_event['PFcluster0Id'] == noise_id)
                flat_event['pf_hit'] = 0
                flat_event.loc[~pf_noise_mask, 'pf_hit'] = 1
                flat_event.to_pickle(raw_data_dir / f'event_{n+ni:05}.pkl')
            ni += n + 1