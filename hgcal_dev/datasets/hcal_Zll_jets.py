from dataclasses import dataclass

from .calo import CaloDataModule, CaloDataset


class HCalZllJetsDataset(CaloDataset):
    def __init__(self, files, voxel_size, task, scale, std, mean, feats, coords, instance_label):
        if instance_label == 'truth':
            instance_label = 'trackId'
        elif instance_label == 'antikt':
            instance_label = 'RHAntiKtCluster_reco'
        elif instance_label == 'pf':
            instance_label = 'PFcluster0Id'
        else:
            raise ValueError()
        semantic_label = 'hit'
        weight = 'energy'
        super().__init__(files, voxel_size, task, scale, std, mean,
                         feats, coords, weight, semantic_label, instance_label)


@dataclass
class HCalZllJetsDataModule(CaloDataModule):
    def make_dataset(self, files) -> HCalZllJetsDataset:
        return HCalZllJetsDataset(files, self.voxel_size, self.task, self.scale, self.std, self.mean, self.feats, self.coords, self.instance_label)
