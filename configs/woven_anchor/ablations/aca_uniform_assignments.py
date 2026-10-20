from configs.woven_anchor.ablations.base import cfg as base_cfg

cfg = base_cfg.clone()
cfg.WOVEN_ANCHOR.MODULE.ANCHOR.ASSIGNMENT_MODE = "uniform"
