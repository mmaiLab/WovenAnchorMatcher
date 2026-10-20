from src.config.default import _CN as cfg

# pose estimation
cfg.TRAINER.POSE_ESTIMATION_METHOD = 'LO-RANSAC'
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5

cfg.TRAINER.EPI_ERR_THR = 5e-4  # recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue)

cfg.WOVEN_ANCHOR.MP = False
cfg.WOVEN_ANCHOR.EVAL_TIMES = 1
cfg.WOVEN_ANCHOR.COARSE.INFERENCE = True
cfg.WOVEN_ANCHOR.FINE.INFERENCE = True
cfg.WOVEN_ANCHOR.COARSE.USE_SM = True
cfg.WOVEN_ANCHOR.COARSE.THR = 0.2
cfg.WOVEN_ANCHOR.FINE.THR = 0.1

cfg.WOVEN_ANCHOR.COARSE.BORDER_RM = 2
