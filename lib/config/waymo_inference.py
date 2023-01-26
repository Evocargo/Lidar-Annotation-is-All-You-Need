from yacs.config import CfgNode as CN

_C = CN()

_C.LOG_DIR = 'inference_results/'
_C.GPUS = [0]     
_C.WORKERS = 4
_C.PIN_MEMORY = True
_C.DEBUG = False
_C.DEBUG_N_BATCHES = 0
_C.CLEARML_LOGGING = False
_C.num_seg_class = 2

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# Common params for NETWORK
_C.MODEL = CN(new_allowed=True)
_C.MODEL.NAME = ''
_C.MODEL.WEIGHTS = 'runs/waymo_PSPNET_exps/_2023-01-25-14-10/epoch-300.pth'
_C.MODEL.IMAGE_SIZE = [640, 640] 
_C.MODEL.EXTRA = CN(new_allowed=True)
_C.MODEL.SEGM_CLASSES = 'road'

# Loss params
_C.LOSS = CN(new_allowed=True)
_C.LOSS.LOSS_NAME = ''
_C.LOSS.MULTI_HEAD_LAMBDA = None
_C.LOSS.FL_GAMMA = 0.0  # focal loss gamma
_C.LOSS.CLS_POS_WEIGHT = 1.0  # classification loss positive weights
_C.LOSS.OBJ_POS_WEIGHT = 1.0  # object loss positive weights
_C.LOSS.SEG_POS_WEIGHT = 1.0  # segmentation loss positive weights
_C.LOSS.DA_SEG_GAIN = 1.0  # driving area segmentation loss gain TRY 1.0
_C.LOSS.MASKED = True

# DATASET related params
_C.DATASET = CN(new_allowed=True)
_C.DATASET.DATASET = 'waymo3d_exps' # aka name of the folder to save
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'val'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.SELECT_DATA = False
_C.DATASET.USE_DET_CACHE = True

# Train
_C.TRAIN = CN(new_allowed=True) 
_C.TRAIN.SAVE_LOCALLY_PER_BATCH = False # False for test.py config

# Inference and vis
_C.TEST = CN(new_allowed=True)
_C.TEST.BATCH_SIZE = 1
_C.TEST.PLOTS = True
_C.inference_visualization = True
_C.save_gt = False
_C.save_video = False
_C.dataset_split = 'val' #'test'

def update_config(cfg, args):
    cfg.defrost()
        
    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights

    if args.inference_visualization:
        cfg.inference_visualization = args.inference_visualization
        cfg.TRAIN.SAVE_LOCALLY_PER_BATCH = False

    if args.save_video:
        cfg.save_video = args.save_video
    
    if args.save_gt:
        cfg.save_gt = args.save_gt
    
    try: 
        cfg.imgs_list
    except:
        cfg.imgs_list = None

    cfg.freeze()
