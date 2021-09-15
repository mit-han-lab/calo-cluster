from omegaconf import DictConfig, OmegaConf, open_dict


def fix_config(cfg: DictConfig) -> None:
    with open_dict(cfg):
        semantic = 'semantic_criterion' in cfg
        instance = 'embed_criterion' in cfg
        if semantic and instance:
            cfg.criterion.task = 'panoptic'
        elif semantic:
            cfg.criterion.task = 'semantic'
        elif instance:
            cfg.criterion.task = 'instance'
        else:
            raise RuntimeError('semantic_criterion and/or embed_criterion must be set!')

        cfg.dataset.task = cfg.criterion.task

        if instance:
            requires_semantic = ('method' not in cfg.embed_criterion) or cfg.embed_criterion.method in ['ignore', 'separate']
            cfg.criterion.requires_semantic = requires_semantic
            if requires_semantic:
                cfg.dataset.task = 'panoptic'
