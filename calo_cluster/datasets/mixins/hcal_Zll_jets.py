from ..pandas import PandasDataset

class HCalZllJetsMixin(PandasDataset):
    def __init__(self, instance_target, **kwargs):
        if instance_target == 'truth':
            semantic_label = 'hit'
            instance_label = 'trackId'
        elif instance_target == 'antikt':
            raise NotImplementedError()
            #instance_label = 'RHAntiKtCluster_reco'
        elif instance_target == 'pf':
            semantic_label = 'pf_hit'
            instance_label = 'PFcluster0Id'
        else:
            raise RuntimeError()
        super().__init__(semantic_label=semantic_label, instance_label=instance_label, weight='energy', **kwargs)