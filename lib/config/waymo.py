from yacs.config import CfgNode as CN

_C = CN()

_C.LOG_DIR = 'runs/'
_C.GPUS = [0]     
_C.WORKERS = 4
_C.PIN_MEMORY = True
_C.PRINT_FREQ = 20
_C.DEBUG = False
_C.DEBUG_N_BATCHES = 0
_C.num_seg_class = 2

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN(new_allowed=True)
_C.MODEL.NAME = ''
_C.MODEL.IMAGE_SIZE = [640, 640]  # width * height, ex: 192 * 256
_C.MODEL.SEGM_CLASSES = 'road'

# loss params
_C.LOSS = CN(new_allowed=True)
_C.LOSS.LOSS_NAME = ''
_C.LOSS.MULTI_HEAD_LAMBDA = None
_C.LOSS.FL_GAMMA = 0.0  # focal loss gamma
_C.LOSS.CLS_POS_WEIGHT = 1.0  # classification loss positive weights
_C.LOSS.OBJ_POS_WEIGHT = 1.0  # object loss positive weights
_C.LOSS.SEG_POS_WEIGHT = 1.0  # segmentation loss positive weights
_C.LOSS.DA_SEG_GAIN = 1.0  # driving area segmentation loss gain CHANHED FROM 0.2
_C.LOSS.MASKED = True # for lidar based masked loss

# DATASET related params
_C.DATASET = CN(new_allowed=True)
_C.DATASET.DATASET = 'waymo_PSPNET_exps' # aka name of the folder to save
_C.DATASET.DATASETS_FRACTIONS = [1., 1.] # youtube_beta, bags_historical, side, bdd 
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'val'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.ORG_IMG_SIZE = [720, 1280]
_C.DATASET.USE_DET_CACHE = True
_C.DATASET.WAYMO_DILATION = False
_C.DATASET.MASKS_ONLY = False # 1
_C.DATASET.LIDAR_DATA_ONLY = False # 2 ## if 1 and 2 are False --> mixing 

# training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 10
_C.DATASET.TRANSLATE = 0.1
_C.DATASET.SHEAR = 0.0
_C.DATASET.COLOR_RGB = False
_C.DATASET.HSV_H = 0.015  # image HSV-Hue augmentation (fraction)
_C.DATASET.HSV_S = 0.7  # image HSV-Saturation augmentation (fraction)
_C.DATASET.HSV_V = 0.4  # image HSV-Value augmentation (fraction)

# train
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
_C.TRAIN.BATCH_SIZE_PER_GPU = 6
_C.TRAIN.SHUFFLE = True

_C.TRAIN.PLOT = True
_C.TRAIN.SAVE_LOCALLY = True
_C.TRAIN.CLEARML_LOGGING = False

# testing
_C.TEST = CN(new_allowed=True)
_C.TEST.BATCH_SIZE_PER_GPU = 6
_C.TEST.MODEL_FILE = ''
_C.TEST.SAVE_TXT = False
_C.TEST.PLOTS = True

# inference and vis
_C.inference_visualization = False
_C.save_video = False
_C.vis_train_gt = True

def update_config(cfg, args):
    cfg.defrost()

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    cfg.freeze()
