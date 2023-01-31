import argparse
import os, sys
import math
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import pprint
import torch
import torch.nn.parallel
from torch.cuda import amp
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms
import numpy as np

from tensorboardX import SummaryWriter
from lib.dataset.WaymoSegmDataset import WaymoSegmDataset
from lib.dataset.Waymo2dSegmDataset import Waymo2dSegmDataset
from lib.config.waymo import _C as cfg
from lib.config.waymo import update_config
from lib.core.loss import get_loss
from lib.core.function import train, validate
from lib.utils.utils import get_optimizer, save_checkpoint, create_logger
from lib.utils.dataloader import WeightedDataLoader
from lib.utils import DataLoaderX

# ClearML
from clearml import Logger # TO FIX

# PSPNet
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import segmentation_models_pytorch as smp

def parse_args():
    parser = argparse.ArgumentParser(description='Train Multitask network')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='runs/')
    args = parser.parse_args()
    return args


def main():
    # set all the configurations
    args = parse_args()
    update_config(cfg, args)

    # Set DDP variables
    global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

    rank = global_rank
    logger, final_output_dir, tb_log_dir = create_logger(cfg, cfg.LOG_DIR, 'train')
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    device = torch.device('cuda:0')
    
    print("MODEL")
    model = smp.PSPNet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=2,                      # model output channels (number of classes in your dataset)
        activation='sigmoid',
    ).cuda()

    # define loss criterion and optimizer
    criterion = get_loss(cfg, device=device)
    optimizer = get_optimizer(cfg, model)
    lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * \
                   (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF  # cosine
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH

    print("TRAIN: begin to load data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if cfg.DATASET.MASKS_ONLY:
        print(f'We will use only 2d segmentation masks')
        dataset1 = Waymo2dSegmDataset(
            cfg=cfg,
            is_train=True,
            inputsize=cfg.MODEL.IMAGE_SIZE,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]),
            data_path="/mnt/large/data/waymo_2d_3d_segm", # /mnt/large/data/waymo_segm
            split='train',
        )
        dataset1.name = "Waymo Segmentation 2d"
        datasets = [dataset1]
        datasets_fractions = [1.]
    elif cfg.DATASET.LIDAR_DATA_ONLY:
        print(f'We will use only reprojected lidar segmentation masks')
        dataset1 = WaymoSegmDataset(
            cfg=cfg,
            is_train=True,
            inputsize=cfg.MODEL.IMAGE_SIZE,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]),
            data_path="/mnt/large/data/waymo_2d_3d_segm", # /mnt/large/data/waymo_segm
            split='train',
        )
        dataset1.name = "Waymo Segmentation repojected 3d"
        datasets = [dataset1]
        datasets_fractions = [1.]
    else:
        print(f'We will use both 2d masks and reprojected 3d data')
        dataset1 = WaymoSegmDataset(
            cfg=cfg,
            is_train=True,
            inputsize=cfg.MODEL.IMAGE_SIZE,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]),
            data_path="/mnt/large/data/waymo_2d_3d_segm", # /mnt/large/data/waymo_segm
            split='train',
            from_img=0, # TO FIX hardcoded for correct mixing
            to_img=926,
        )
        dataset1.name = "Waymo Segmentation repojected 3d"

        dataset2 = Waymo2dSegmDataset(
            cfg=cfg,
            is_train=True,
            inputsize=cfg.MODEL.IMAGE_SIZE,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]),
            data_path="/mnt/large/data/waymo_2d_3d_segm/",
            split='train',
            from_img=926, 
            to_img=1852,
        )
        dataset2.name = "Waymo Segmentation 2d"
        datasets = [dataset1, dataset2]
        datasets_fractions = cfg.DATASET.DATASETS_FRACTIONS

    dataset_samples_list = []
    num_samples = 0
    for dataset_index, dataset_ in enumerate(datasets):
        datasets_fraction = datasets_fractions[dataset_index]
        dataset_samples = int(datasets_fraction * len(dataset_))
        dataset_samples_list.append(dataset_samples)
        num_samples += dataset_samples
        
        print(
            f"Dataset {dataset_.name} with fraction {datasets_fraction} is created"
        )
        print(f"{dataset_samples} samples will be used for training")

    weights = np.array(dataset_samples_list) / num_samples
    concat_dataset = ConcatDataset(datasets)
    
    train_loader = WeightedDataLoader(
        concat_dataset, 
        weights=weights, 
        num_samples=num_samples, 
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=Waymo2dSegmDataset.collate_fn,
    )
    num_batch = len(train_loader)
    print('TRAIN: load data finished')
    
    # VAL data loading
    print("VAL: begin to load data")
    valid_dataset = Waymo2dSegmDataset(
        cfg=cfg,
        is_train=False,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        data_path="/mnt/large/data/waymo_2d_3d_segm", # /mnt/large/data/waymo_segm
        split='val',
    )

    valid_loader = DataLoaderX(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=Waymo2dSegmDataset.collate_fn,
    )
    print('VAL: load data finished')

    # training
    num_warmup = max(round(cfg.TRAIN.WARMUP_EPOCHS * num_batch), 1000)
    scaler = amp.GradScaler(enabled=device.type != 'cpu')
    print('=> start training...')
    for epoch in range(begin_epoch+1, cfg.TRAIN.END_EPOCH+1):
        if rank != -1:
            train_loader.sampler.set_epoch(epoch)
        # train for one epoch
        train(cfg, train_loader, model, criterion, optimizer, scaler,
              epoch, num_batch, num_warmup, writer_dict, logger, device, final_output_dir, Logger)
        
        lr_scheduler.step()

        # evaluate on validation set
        if (epoch % cfg.TRAIN.VAL_FREQ == 0 or epoch == cfg.TRAIN.END_EPOCH):
            da_segment_results, total_loss = validate(
                epoch, cfg, valid_loader, model, criterion,
                final_output_dir, device, clearml_logger=Logger,
            )

            msg = 'Epoch: [{0}]    Loss({loss:.3f})\n' \
                      'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n'.format(
                          epoch, loss=total_loss, 
                          da_seg_acc=da_segment_results[0], da_seg_iou=da_segment_results[1], 
                          da_seg_miou=da_segment_results[2])
            logger.info(msg)
        
        # save checkpoint model and best model
        if epoch % 50 == 0:
            savepath = os.path.join(final_output_dir, f'epoch-{epoch}.pth')
            logger.info('=> saving checkpoint to {}'.format(savepath))
            save_checkpoint(
                epoch=epoch,
                name=cfg.MODEL.NAME,
                model=model,
                optimizer=optimizer,
                output_dir=final_output_dir,
                filename=f'epoch-{epoch}.pth'
            )

    final_model_state_file = os.path.join(final_output_dir, 'final_state.pth')
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    model_state = model.state_dict()
    torch.save(model_state, final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()