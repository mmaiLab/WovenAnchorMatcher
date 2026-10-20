from yacs.config import CfgNode as CN

cfg = CN()
cfg.WOVEN_ANCHOR = CN()
cfg.WOVEN_ANCHOR.MODULE = CN()

# Two stages of intra-image aggregation, inter-image aggregation, and an FFN.
cfg.WOVEN_ANCHOR.MODULE.LAYERS = [
    "wovenmamba", "anchorcrossattention", "ffnblock",
    "wovenmamba", "anchorcrossattention", "ffnblock",
]

# WovenMamba: scan composition and state-space parameters.
cfg.WOVEN_ANCHOR.MODULE.WOVEN = CN()
cfg.WOVEN_ANCHOR.MODULE.WOVEN.D_STATE = 8
cfg.WOVEN_ANCHOR.MODULE.WOVEN.D_CONV = 3
cfg.WOVEN_ANCHOR.MODULE.WOVEN.EXPAND = 2
cfg.WOVEN_ANCHOR.MODULE.WOVEN.USE_SEPARATOR = True
cfg.WOVEN_ANCHOR.MODULE.WOVEN.USE_CHECKPOINT = True

# AnchorCrossAttention: window grouping, descriptors, and attention.
cfg.WOVEN_ANCHOR.MODULE.ANCHOR = CN()
cfg.WOVEN_ANCHOR.MODULE.ANCHOR.N_ANCHORS = 4
cfg.WOVEN_ANCHOR.MODULE.ANCHOR.DIM_ANCHOR = 256
cfg.WOVEN_ANCHOR.MODULE.ANCHOR.WINDOW_SIZE = 8
cfg.WOVEN_ANCHOR.MODULE.ANCHOR.USE_OVERLAP = True  # Half-window stride.
cfg.WOVEN_ANCHOR.MODULE.ANCHOR.N_HEADS = 4
cfg.WOVEN_ANCHOR.MODULE.ANCHOR.USE_ANC_NORM = False
cfg.WOVEN_ANCHOR.MODULE.ANCHOR.USE_F_RAW = True
cfg.WOVEN_ANCHOR.MODULE.ANCHOR.USE_F_NORM = True
cfg.WOVEN_ANCHOR.MODULE.ANCHOR.USE_F_GEO = True
cfg.WOVEN_ANCHOR.MODULE.ANCHOR.DIM_GEO = 16
cfg.WOVEN_ANCHOR.MODULE.ANCHOR.GEO_MODE = "stats"
cfg.WOVEN_ANCHOR.MODULE.ANCHOR.ASSIGNMENT_MODE = "learned"
cfg.WOVEN_ANCHOR.MODULE.ANCHOR.ATTENTION_SCOPE = "per_anchor"
cfg.WOVEN_ANCHOR.MODULE.ANCHOR.OVERLAP_WEIGHT_MODE = "entropy"
