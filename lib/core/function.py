import time
import torch
import numpy as np
from pathlib import Path
import cv2
import os
import math
from torch.cuda import amp
from tqdm import tqdm

from lib.core.evaluate import SegmentationMetric
from lib.utils import show_seg_result, inverse_normalize, AverageMeter
from evopy.images import write_video # TO FIX


def train(config, train_loader, model, criterion, optimizer, scaler, epoch, num_batch, num_warmup,
          writer_dict, logger, device, final_output_dir=None, clearml_logger=None):
    """
    TODO
    """ 
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    seg_da_loss = AverageMeter()

    if final_output_dir:
        save_dir = final_output_dir + os.path.sep + 'visualization'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    # switch to train mode
    model.train()
    
    start = time.time()
    for batch_i, (input, target, _paths, shapes) in enumerate(train_loader):
        num_iter = batch_i + num_batch * (epoch - 1)

        if num_iter < num_warmup:
            # warm up
            lf = lambda x: ((1 + math.cos(x * math.pi / config.TRAIN.END_EPOCH)) / 2) * \
                           (1 - config.TRAIN.LRF) + config.TRAIN.LRF  # cosine
            xi = [0, num_warmup]
            for j, x in enumerate(optimizer.param_groups):
                x['lr'] = np.interp(num_iter, xi, [config.TRAIN.WARMUP_BIASE_LR if j == 2 else 0.0, 
                                                   x['initial_lr'] * lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(num_iter, xi, [config.TRAIN.WARMUP_MOMENTUM, 
                                                             config.TRAIN.MOMENTUM])

        data_time.update(time.time() - start)
        if not config.DEBUG:
            input = input.to(device, non_blocking=True)
            assign_target = []
            for tgt in target:
                assign_target.append(tgt.to(device))
            target = assign_target
        with amp.autocast(enabled=device.type != 'cpu'):
            outputs = model(input)
            total_loss, lseg_da = criterion(outputs, target, shapes)
        
        # compute gradient and do update step
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if config.vis_train_gt:
            if batch_i in [0, 1, 2]:
                image_ind = 1 # just the first image from three diferent batches
                img_inv_norm = inverse_normalize(input[image_ind], mean=(0.485, 0.456, 0.406), 
                                                 std=(0.229, 0.224, 0.225))
                img_test = cv2.cvtColor(img_inv_norm.cpu().numpy().transpose(1, 2, 0) * 255, 
                                        cv2.COLOR_BGR2RGB)
                da_gt_mask = target[1][image_ind].unsqueeze(0)
                _, da_gt_mask = torch.max(da_gt_mask, 1)
                da_gt_mask = da_gt_mask.int().squeeze().cpu().numpy()
                img_test1 = img_test.copy()
                _img_with_gt = show_seg_result(img_test1, da_gt_mask, batch_i, epoch, 
                                                save_dir, is_gt=True, config=config, 
                                                clearml_logger=clearml_logger, prefix='train_')

        # measure accuracy and record loss
        input_size = input.size(0)
        losses.update(total_loss.item(), input_size)
        seg_da_loss.update(lseg_da, input_size)

        # measure elapsed time
        batch_time.update(time.time() - start)
        if batch_i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                    'Speed {speed:.1f} samples/s\t' \
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                    'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                        epoch, batch_i, len(train_loader), batch_time=batch_time,
                        speed=input.size(0)/batch_time.val,
                        data_time=data_time, loss=losses)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

        # BATCH END
        if config.DEBUG_N_BATCHES > 0 and batch_i == config.DEBUG_N_BATCHES:
            break

    if config.CLEARML_LOGGING:
        clearml_logger.current_logger().report_scalar(title="TRAIN_LOSS", series="TRAIN_LOSS", 
                                                      value=losses.avg, iteration=epoch)
        clearml_logger.current_logger().report_scalar(title="LOSSES", series="seg_da_LOSS", 
                                                      value=seg_da_loss.avg, iteration=epoch)


def validate(epoch, config, val_loader, model, criterion, output_dir, 
             device='cpu', clearml_logger=None, half=False):
    """
    TODO
    """
    # setting
    save_dir = output_dir + os.path.sep + 'visualization'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    test_batch_size = config.TEST.BATCH_SIZE
    da_metric = SegmentationMetric(config.num_seg_class)
    
    losses = AverageMeter()
    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()
    da_mIoU_seg = AverageMeter()

    # switch to train mode
    model.eval()
    res_images = []
    
    for batch_i, (img, target, paths, shapes) in tqdm(enumerate(val_loader), total=len(val_loader)):
        if not config.DEBUG:
            img = img.to(device, non_blocking=True)
            if half:
                img = img.half()
            assign_target = []
            for tgt in target:
                assign_target.append(tgt.to(device))
            target = assign_target
            nb, _, height, width = img.shape    #batch size, channel, height, width
        
        with torch.no_grad():
            pad_w, pad_h = shapes[0][1][1]
            pad_w = int(pad_w)
            pad_h = int(pad_h)

            da_seg_out = model(img)

            # driving area segment evaluation
            _, da_predict = torch.max(da_seg_out, 1)
            _, da_gt = torch.max(target[1], 1)
            da_predict = da_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
            da_gt = da_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]
            
            da_metric.reset()
            da_metric.addBatch(da_predict.cpu(), da_gt.cpu())
            da_acc = da_metric.pixelAccuracy()
            da_IoU = da_metric.IntersectionOverUnion()
            da_mIoU = da_metric.meanIntersectionOverUnion()

            da_acc_seg.update(da_acc,img.size(0))
            da_IoU_seg.update(da_IoU,img.size(0))
            da_mIoU_seg.update(da_mIoU,img.size(0))
            
            # compute loss
            total_loss, _lseg_da = criterion(da_seg_out, target, shapes)   
            losses.update(total_loss.item(), img.size(0))

            if config.TEST.PLOTS and not config.inference_visualization:
                if batch_i == 0 or batch_i == 1 or batch_i == 2: # TO FIX 
                    image_ind = 0 # just the first image from three diferent batches
                    img_inv_norm = inverse_normalize(img[image_ind], mean=(0.485, 0.456, 0.406), 
                                                     std=(0.229, 0.224, 0.225))
                    img_test = cv2.cvtColor(img_inv_norm.cpu().numpy().transpose(1, 2, 0) * 255, 
                                            cv2.COLOR_BGR2RGB)
                    
                    da_seg_mask = da_seg_out[image_ind].unsqueeze(0)
                    _, da_seg_mask = torch.max(da_seg_mask, 1) # if the first tensor (not road) max then 0, else 1 which is road
                    da_gt_mask = target[1][image_ind].unsqueeze(0)
                    _, da_gt_mask = torch.max(da_gt_mask, 1)
                    da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
                    da_gt_mask = da_gt_mask.int().squeeze().cpu().numpy()

                    img_test1 = img_test.copy()
                    _img_with_segm = show_seg_result(img_test, da_seg_mask, batch_i, epoch, 
                                                     save_dir, config=config, 
                                                     clearml_logger=clearml_logger)
                    _img_with_gt = show_seg_result(img_test1, da_gt_mask, batch_i, epoch, 
                                                   save_dir, is_gt=True, config=config, 
                                                   clearml_logger=clearml_logger)
            else:
                for image_ind in range(nb): # for each image in batch
                    folder_to_save = Path(f"{save_dir}/inference_results/")
                    folder_to_save.mkdir(parents=True, exist_ok=True)

                    if config.save_gt:
                        folder_to_save_gt = Path(f"{save_dir}/gt/") 
                        folder_to_save_gt.mkdir(parents=True, exist_ok=True)

                    if hasattr(val_loader.dataset, 'data_path'):
                        filename = val_loader.dataset.data_path((batch_i * test_batch_size) + image_ind).name
                    else:
                        filename = f"{batch_i}_{image_ind}_det_pred.jpg"

                    img_inv_norm = inverse_normalize(img[image_ind], mean=(0.485, 0.456, 0.406), 
                                                     std=(0.229, 0.224, 0.225))
                    img_test = cv2.cvtColor(img_inv_norm.cpu().numpy().transpose(1, 2, 0) * 255, 
                                            cv2.COLOR_BGR2RGB)

                    da_seg_mask = da_seg_out[image_ind].unsqueeze(0)
                    _, da_seg_mask = torch.max(da_seg_mask, 1)
                    da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()

                    img_test1 = img_test.copy()
                    img_with_predict = show_seg_result(img_test1, da_seg_mask, batch_i, epoch, 
                                                       save_dir, config=config, 
                                                       clearml_logger=clearml_logger)
                    cv2.imwrite(f"{folder_to_save}/{filename}", img_with_predict)
                    
                    if config.save_gt:
                        da_gt_mask = target[1][image_ind].unsqueeze(0)
                        _, da_gt_mask = torch.max(da_gt_mask, 1)
                        da_gt_mask = da_gt_mask.int().squeeze().cpu().numpy()
                        img_test2 = img_test.copy()
                        img_with_gt = show_seg_result(img_test2, da_gt_mask, batch_i, epoch, 
                                                      save_dir, is_gt=True, config=config, 
                                                      clearml_logger=clearml_logger)
                        cv2.imwrite(f"{folder_to_save_gt}/{filename}", img_with_gt)                    
                    
                if config.save_video:
                    res_images.append(cv2.cvtColor(np.array(img_with_predict, dtype=np.uint8), 
                                                   cv2.COLOR_BGR2RGB))
            
        # BATCH END
        if config.DEBUG_N_BATCHES > 0 and batch_i == config.DEBUG_N_BATCHES:
            break
    
    if config.save_video:
        video_path = f"{save_dir}/segm_video_{val_loader.dataset.split}.mp4"
        write_video(res_images, video_path=video_path)

    model.float()  # for training
    da_segment_result = (da_acc_seg.avg, da_IoU_seg.avg, da_mIoU_seg.avg)

    return da_segment_result, losses.avg

