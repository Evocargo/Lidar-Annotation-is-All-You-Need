from yacs.config import CfgNode as CN

_C = CN()

_C.LOG_DIR = 'runs/'
_C.GPUS = [0]     
_C.WORKERS = 4
_C.PIN_MEMORY = False
_C.PRINT_FREQ = 20
_C.DEBUG = False
_C.DEBUG_N_BATCHES = 0
_C.CLEARML_LOGGING = False
_C.num_seg_class = 2

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# Common params for MODEL
_C.MODEL = CN(new_allowed=True)
_C.MODEL.NAME = ''
_C.MODEL.IMAGE_SIZE = [640, 640]  # width * height, ex: 192 * 256
_C.MODEL.SEGM_CLASSES = 'road'

# Loss params
_C.LOSS = CN(new_allowed=True)
_C.LOSS.LOSS_NAME = ''
_C.LOSS.MULTI_HEAD_LAMBDA = None
_C.LOSS.FL_GAMMA = 0.0
_C.LOSS.CLS_POS_WEIGHT = 1.0
_C.LOSS.OBJ_POS_WEIGHT = 1.0
_C.LOSS.SEG_POS_WEIGHT = 1.0
_C.LOSS.DA_SEG_GAIN = 1.0  # driving area segmentation loss gain
_C.LOSS.MASKED = True # for lidar data based masked loss

# DATASET related params
_C.DATASET = CN(new_allowed=True)
_C.DATASET.DATASET = 'kitti_360_PSPNET_exps'
_C.DATASET.PATH = '/hdd2/kitti_360_paper_dataset/'
_C.DATASET.VAL_PATH = '/hdd2/kitti_360_paper_dataset/'
_C.DATASET.DATASETS_FRACTIONS = [1.0, 1.0]
_C.DATASET.DATA_FORMAT = 'png'
_C.DATASET.AUTO_SHAPE = True
_C.DATASET.FILL_BETWEEN_POINTS = False
_C.DATASET.USE_DET_CACHE = True
_C.DATASET.WAYMO_DILATION = False
_C.DATASET.MASKS_ONLY = False # 1
_C.DATASET.LIDAR_DATA_ONLY = False # 2 ## if 1 and 2 are False --> mixing

# "waymo with intersection" mixing only
_C.DATASET.from_img_3D = None
_C.DATASET.to_img_3D = None
_C.DATASET.from_img_2D = None
_C.DATASET.to_img_3D = None

# Training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 10
_C.DATASET.TRANSLATE = 0.1
_C.DATASET.SHEAR = 0.0
_C.DATASET.COLOR_RGB = False
_C.DATASET.HSV_H = 0.015  # image HSV-Hue augmentation (fraction)
_C.DATASET.HSV_S = 0.7  # image HSV-Saturation augmentation (fraction)
_C.DATASET.HSV_V = 0.4  # image HSV-Value augmentation (fraction)

# Train
_C.TRAIN = CN(new_allowed=True)
_C.TRAIN.LR0 = 0.001  # initial learning rate (SGD=1E-2, Adam=1E-3) 0.001
_C.TRAIN.LRF = 0.5  # final OneCycleLR learning rate (lr0 * lrf)
_C.TRAIN.WARMUP_EPOCHS = 3.0
_C.TRAIN.WARMUP_BIASE_LR = 0.1
_C.TRAIN.WARMUP_MOMENTUM = 0.8
_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.937
_C.TRAIN.WD = 0.0005
_C.TRAIN.NESTEROV = True
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 300
_C.TRAIN.VAL_FREQ = 1
_C.TRAIN.BATCH_SIZE = 20
_C.TRAIN.SHUFFLE = True # TO FIX
_C.TRAIN.SAVE_LOCALLY_PER_BATCH = True

# Testing
_C.TEST = CN(new_allowed=True)
_C.TEST.BATCH_SIZE = 1
_C.TEST.PLOTS = True

# Inference and vis
_C.inference_visualization = True
_C.save_video = False
_C.save_gt = True
_C.vis_train_gt = True


def update_config(cfg, args):
    cfg.defrost()

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    cfg.freeze()
