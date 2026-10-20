# Adapted through JamMa from the LoFTR framework; modified for WovenAnchor.
# Inherited MIT/Apache-2.0 terms and attribution: THIRD_PARTY_NOTICES.md.

from yacs.config import CfgNode as CN
_CN = CN()

##############  ↓  WOVEN_ANCHOR Pipeline  ↓  ##############
_CN.WOVEN_ANCHOR = CN()
_CN.WOVEN_ANCHOR.RESOLUTION = (8, 2)
_CN.WOVEN_ANCHOR.FINE_WINDOW_SIZE = 5  # window_size in fine_level, must be odd

_CN.BACKBONE = "ConvNextv2"

_CN.WOVEN_ANCHOR.COARSE = CN()
_CN.WOVEN_ANCHOR.COARSE.D_MODEL = 256
_CN.WOVEN_ANCHOR.COARSE.USE_SM = True
_CN.WOVEN_ANCHOR.COARSE.THR = 0.2
_CN.WOVEN_ANCHOR.COARSE.BORDER_RM = 2
_CN.WOVEN_ANCHOR.COARSE.DSMAX_TEMPERATURE = 0.1
_CN.WOVEN_ANCHOR.COARSE.SKH_ITERS = 3
_CN.WOVEN_ANCHOR.COARSE.TRAIN_COARSE_PERCENT = 0.2  # training tricks: save GPU memory
_CN.WOVEN_ANCHOR.COARSE.TRAIN_PAD_NUM_GT_MIN = 200  # training tricks: avoid DDP deadlock
_CN.WOVEN_ANCHOR.COARSE.INFERENCE = False

_CN.WOVEN_ANCHOR.FINE = CN()
_CN.WOVEN_ANCHOR.FINE.D_MODEL = 64
_CN.WOVEN_ANCHOR.FINE.D_MODEL_M = 128
_CN.WOVEN_ANCHOR.FINE.DENSER = False # if true, match all features in fine-level windows
_CN.WOVEN_ANCHOR.FINE.INFERENCE = False
_CN.WOVEN_ANCHOR.FINE.DSMAX_TEMPERATURE = 0.1
_CN.WOVEN_ANCHOR.FINE.THR = 0.1

_CN.WOVEN_ANCHOR.MODULE = CN()
_CN.WOVEN_ANCHOR.MODULE.LAYERS = [
    "wovenmamba",
    "anchorcrossattention",
    "ffnblock",
    "wovenmamba",
    "anchorcrossattention",
    "ffnblock",
]
_CN.WOVEN_ANCHOR.MODULE.WOVEN = CN()
_CN.WOVEN_ANCHOR.MODULE.ANCHOR = CN()

_CN.WOVEN_ANCHOR.MODULE.WOVEN.D_STATE = 8
_CN.WOVEN_ANCHOR.MODULE.WOVEN.D_CONV = 3
_CN.WOVEN_ANCHOR.MODULE.WOVEN.EXPAND = 2
_CN.WOVEN_ANCHOR.MODULE.WOVEN.USE_SEPARATOR = True
_CN.WOVEN_ANCHOR.MODULE.WOVEN.USE_CHECKPOINT = True

_CN.WOVEN_ANCHOR.MODULE.ANCHOR.N_ANCHORS = 4
_CN.WOVEN_ANCHOR.MODULE.ANCHOR.DIM_ANCHOR = 256
_CN.WOVEN_ANCHOR.MODULE.ANCHOR.WINDOW_SIZE = 8
_CN.WOVEN_ANCHOR.MODULE.ANCHOR.USE_OVERLAP = True
_CN.WOVEN_ANCHOR.MODULE.ANCHOR.N_HEADS = 4
_CN.WOVEN_ANCHOR.MODULE.ANCHOR.USE_ANC_NORM = False
_CN.WOVEN_ANCHOR.MODULE.ANCHOR.USE_F_RAW = True
_CN.WOVEN_ANCHOR.MODULE.ANCHOR.USE_F_NORM = True
_CN.WOVEN_ANCHOR.MODULE.ANCHOR.USE_F_GEO = True
_CN.WOVEN_ANCHOR.MODULE.ANCHOR.DIM_GEO = 16
_CN.WOVEN_ANCHOR.MODULE.ANCHOR.GEO_MODE = "stats"
_CN.WOVEN_ANCHOR.MODULE.ANCHOR.ASSIGNMENT_MODE = "learned"
_CN.WOVEN_ANCHOR.MODULE.ANCHOR.ATTENTION_SCOPE = "per_anchor"  # Within each anchor-ID group.
_CN.WOVEN_ANCHOR.MODULE.ANCHOR.OVERLAP_WEIGHT_MODE = "entropy"


_CN.WOVEN_ANCHOR.LOSS = CN()
_CN.WOVEN_ANCHOR.LOSS.COARSE_WEIGHT = 1.0
_CN.WOVEN_ANCHOR.LOSS.FOCAL_ALPHA = 0.25
_CN.WOVEN_ANCHOR.LOSS.FOCAL_GAMMA = 2.0
_CN.WOVEN_ANCHOR.LOSS.POS_WEIGHT = 1.0
_CN.WOVEN_ANCHOR.LOSS.NEG_WEIGHT = 1.0
_CN.WOVEN_ANCHOR.LOSS.SUB_WEIGHT = 1 * 10**4
_CN.WOVEN_ANCHOR.LOSS.FINE_WEIGHT = 1.0

_CN.WOVEN_ANCHOR.MP = False
_CN.WOVEN_ANCHOR.EVAL_TIMES = 1

##############  Dataset  ##############
_CN.DATASET = CN()
# 1. data config
# training and validating
_CN.DATASET.TRAINVAL_DATA_SOURCE = None  # options: ['ScanNet', 'MegaDepth']
_CN.DATASET.TRAIN_DATA_ROOT = None
_CN.DATASET.TRAIN_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TRAIN_NPZ_ROOT = None
_CN.DATASET.TRAIN_LIST_PATH = None
_CN.DATASET.TRAIN_INTRINSIC_PATH = None
_CN.DATASET.VAL_DATA_ROOT = None
_CN.DATASET.VAL_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.VAL_NPZ_ROOT = None
_CN.DATASET.VAL_LIST_PATH = None    # None if val data from all scenes are bundled into a single npz file
_CN.DATASET.VAL_INTRINSIC_PATH = None
# testing
_CN.DATASET.TEST_DATA_SOURCE = None
_CN.DATASET.TEST_DATA_ROOT = None
_CN.DATASET.TEST_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TEST_NPZ_ROOT = None
_CN.DATASET.TEST_LIST_PATH = None   # None if test data from all scenes are bundled into a single npz file
_CN.DATASET.TEST_INTRINSIC_PATH = None
_CN.DATASET.INPUT_PAIRS = None
_CN.DATASET.NPE_NAME = None

# 2. dataset config
# general options
_CN.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.4  # discard data with overlap_score < min_overlap_score
_CN.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
_CN.DATASET.AUGMENTATION_TYPE = None  # options: [None, 'dark', 'mobile']
_CN.DATASET.CORR_TH = 5

# MegaDepth options
_CN.DATASET.MGDPT_IMG_RESIZE = 832  # resize the longer side, zero-pad bottom-right to square.
_CN.DATASET.MGDPT_IMG_PAD = True  # pad img to square with size = MGDPT_IMG_RESIZE
_CN.DATASET.MGDPT_DEPTH_PAD = True  # pad depthmap to square with size = 2000
_CN.DATASET.MGDPT_DF = 8

# HPatches options
_CN.DATASET.HP_IMG_RESIZE = 480
_CN.DATASET.HP_DF = 8
_CN.DATASET.HP_PAD = True
_CN.DATASET.HP_PAD_DF = 16
_CN.DATASET.RANSAC_THR = 3.0
_CN.DATASET.TOPK = 1000


##############  Trainer  ##############
_CN.TRAINER = CN()
_CN.TRAINER.WORLD_SIZE = 1
_CN.TRAINER.CANONICAL_BS = 5
_CN.TRAINER.CANONICAL_LR = 6e-3
_CN.TRAINER.FIND_LR = False  # use learning rate finder from pytorch-lightning

# optimizer
_CN.TRAINER.OPTIMIZER = "adamw"  # [adam, adamw]
_CN.TRAINER.TRUE_LR = None  # this will be calculated automatically at runtime
_CN.TRAINER.EN_LR = None
_CN.TRAINER.DE_LR = None
_CN.TRAINER.ADAM_DECAY = 0.0
_CN.TRAINER.ADAMW_DECAY = 0.1

# step-based warm-up
_CN.TRAINER.WARMUP_TYPE = 'linear'  # [linear, constant]
_CN.TRAINER.WARMUP_RATIO = 0.
_CN.TRAINER.WARMUP_STEP = 4800

# learning rate scheduler
_CN.TRAINER.SCHEDULER = 'MultiStepLR'  # [MultiStepLR, CosineAnnealing, ExponentialLR]
_CN.TRAINER.SCHEDULER_INTERVAL = 'epoch'    # [epoch, step]
_CN.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12]
_CN.TRAINER.MSLR_GAMMA = 0.5
_CN.TRAINER.COSA_TMAX = 30
_CN.TRAINER.ELR_GAMMA = 0.999992

# plotting related (not used now)
_CN.TRAINER.ENABLE_PLOTTING = False
_CN.TRAINER.N_VAL_PAIRS_TO_PLOT = 32     # number of val/test pairs for plotting
_CN.TRAINER.PLOT_MODE = 'evaluation'  # ['evaluation', 'confidence']
_CN.TRAINER.PLOT_MATCHES_ALPHA = 'dynamic'

# geometric metrics and pose solver
_CN.TRAINER.EPI_ERR_THR = 1e-4  # recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue)
_CN.TRAINER.POSE_GEO_MODEL = 'E'  # ['E', 'F', 'H']
_CN.TRAINER.POSE_ESTIMATION_METHOD = 'LO-RANSAC'
_CN.TRAINER.RANSAC_PIXEL_THR = 0.5
_CN.TRAINER.RANSAC_CONF = 0.99999
_CN.TRAINER.RANSAC_MAX_ITERS = 10000
_CN.TRAINER.USE_MAGSACPP = False

# data sampler for train_dataloader
_CN.TRAINER.DATA_SAMPLER = 'scene_balance'  # options: ['scene_balance', 'random', 'normal']
# 'scene_balance' config
_CN.TRAINER.N_SAMPLES_PER_SUBSET = 200
_CN.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = True  # whether sample each scene with replacement or not
_CN.TRAINER.SB_SUBSET_SHUFFLE = True  # after sampling from scenes, whether shuffle within the epoch or not
_CN.TRAINER.SB_REPEAT = 1  # repeat N times for training the sampled data
# 'random' config
_CN.TRAINER.RDM_REPLACEMENT = True
_CN.TRAINER.RDM_NUM_SAMPLES = None

# gradient clipping
_CN.TRAINER.GRADIENT_CLIPPING = 0.5

# reproducibility
# This seed affects the data sampling. With the same seed, the data sampling is promised
# to be the same. When resume training from a checkpoint, it's better to use a different
# seed, otherwise the sampled data will be exactly the same as before resuming, which will
# cause less unique data items sampled during the entire training.
# Use of different seed values might affect the final training result, since not all data items
# are used during training on ScanNet. (60M pairs of images sampled during traing from 230M pairs in total.)
_CN.TRAINER.SEED = 66


def get_cfg_defaults():
    """Return an independent configuration with the paper WAM model defaults."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
