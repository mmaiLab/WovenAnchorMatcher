from src.config.default import _CN as cfg

cfg.WOVEN_ANCHOR.RESOLUTION = (8, 2)
cfg.WOVEN_ANCHOR.FINE_WINDOW_SIZE = 5  # window_size in fine_level, must be odd
cfg.WOVEN_ANCHOR.COARSE.D_MODEL = 256
cfg.WOVEN_ANCHOR.FINE.D_MODEL = 64

cfg.TRAINER.SCHEDULER = 'CosineAnnealing'  # [MultiStepLR, CosineAnnealing, ExponentialLR]
cfg.TRAINER.SCHEDULER_INTERVAL = 'epoch'
cfg.TRAINER.COSA_TMAX = 30  # train.py sets this to max_epochs (default: 20).
cfg.TRAINER.CANONICAL_BS = 2
cfg.TRAINER.CANONICAL_LR = 1e-4
cfg.TRAINER.WARMUP_STEP = 58200  # from JamMa

# pose estimation
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5

cfg.TRAINER.OPTIMIZER = "adamw"
cfg.TRAINER.ADAMW_DECAY = 0.1
cfg.TRAINER.N_VAL_PAIRS_TO_PLOT = 1

cfg.TRAINER.SEED = 66
