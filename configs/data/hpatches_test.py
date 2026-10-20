from configs.data.base import cfg

cfg.DATASET.TEST_DATA_SOURCE = "HPatches"
cfg.DATASET.TEST_DATA_ROOT = "data/hpatches/hpatches-sequences-release/"

cfg.DATASET.HP_IMG_RESIZE = 480
cfg.DATASET.HP_DF = 8
cfg.DATASET.HP_PAD = True
cfg.DATASET.HP_PAD_DF = 8
cfg.DATASET.TOPK = 1000
cfg.DATASET.RANSAC_THR = 3.0