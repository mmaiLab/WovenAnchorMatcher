from configs.woven_anchor.ablations.base import cfg as base_cfg

cfg = base_cfg.clone()
cfg.WOVEN_ANCHOR.MODULE.LAYERS = ["wovenmamba", "crossfourmamba", "ffnblock", "wovenmamba", "crossfourmamba", "ffnblock"]
