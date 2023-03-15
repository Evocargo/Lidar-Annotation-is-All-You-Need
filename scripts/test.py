import argparse
import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import pprint
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from lib.utils import DataLoaderX
from lib.dataset.Waymo2dSegmDataset import Waymo2dSegmDataset
from lib.config.waymo_inference import _C as cfg
from lib.config.waymo_inference import update_config
from lib.core.loss import get_loss
from lib.core.function import validate
from lib.utils.utils import create_logger
# PSPNet
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import segmentation_models_pytorch as smp

def parse_args():
    parser = argparse.ArgumentParser(description='Test Multitask network')
    parser.add_argument('--weights', type=str, default='', help='model.pth path')
    parser.add_argument('--inference_visualization', type=bool, default=False, help='save images with detection and segmentation results')
    parser.add_argument('--save_video', action='store_true', help='to save video with results')
    parser.add_argument('--save_gt', type=bool, default=False, help='to visualize gt')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, _tb_log_dir = create_logger(cfg, cfg.LOG_DIR, 'test')
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    device = torch.device('cuda:0')
    
    print("MODEL")
    model = smp.PSPNet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=2,                      # model output channels (number of classes in your dataset)
        activation='sigmoid',
    ).to(device)
    print(f"Loading weights: {cfg.MODEL.WEIGHTS}")
    checkpoint = torch.load(cfg.MODEL.WEIGHTS)
    model.load_state_dict(checkpoint['state_dict'])
    print('Bulid model finished')
    
    # define loss function (criterion) and optimizer
    criterion = get_loss(cfg, device=device)

    print("DATA LOAD")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    valid_dataset = Waymo2dSegmDataset(
        cfg=cfg,
        is_train=False,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        data_path="/mnt/large/data/waymo_2d_3d_segm/",
        split=cfg.dataset_split,
    )

    valid_loader = DataLoaderX(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=False,
        collate_fn=Waymo2dSegmDataset.collate_fn,
    )
    print('Load data finished')
    
    epoch = 0 # special for test
    da_segment_results, total_loss = validate(
        epoch, cfg, valid_loader, model, criterion,
        final_output_dir, device=device,
    )
    msg =   'Test:    Loss({loss:.3f})\n' \
            'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n'.format(
                          loss=total_loss, da_seg_acc=da_segment_results[0], 
                          da_seg_iou=da_segment_results[1], da_seg_miou=da_segment_results[2],)
    logger.info(msg)
    print("Test finish")


if __name__ == '__main__':
    main()
    