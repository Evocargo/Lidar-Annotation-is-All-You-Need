import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.cuda import amp
from tqdm import tqdm
from yacs.config import CfgNode

from lib.core.evaluate import SegmentationMetric
from lib.utils import AverageMeter, inverse_normalize, show_seg_result, write_video


def train(
    config: CfgNode,
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    epoch: int,
    num_batch: int,
    num_warmup: int,
    writer_dict: dict,
    logger: Any,
    device: torch.device,
    final_output_dir: Optional[str] = None,
    clearml_logger: Optional[Any] = None,
) -> None:
    """
    Run one training epoch.

    Args:
        config: Configuration object with parameters and hyperparameters.
        train_loader: DataLoader for the training data.
        model: Model to train.
        criterion: Loss function.
        optimizer: Optimizer.
        scaler: Gradient scaler for mixed precision training.
        epoch: Current epoch number.
        num_batch: Number of batches in the dataset.
        num_warmup: Number of warmup iterations.
        writer_dict: Dictionary containing the writer object for logging.
        logger: Logger object for training logging.
        device: Device on which to run the training.
        final_output_dir: Directory where to save visualizations.
        clearml_logger: Logger for ClearML.
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    seg_da_loss = AverageMeter()

    if final_output_dir:
        save_dir = final_output_dir + os.path.sep + "visualization"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    # switch to train mode
    model.train()

    start = time.time()
    for batch_i, (input, target, _paths, shapes) in enumerate(train_loader):
        num_iter = batch_i + num_batch * (epoch - 1)

        if num_iter < num_warmup:
            # warm up
            lf = (
                lambda x: ((1 + math.cos(x * math.pi / config.TRAIN.END_EPOCH)) / 2) * (1 - config.TRAIN.LRF)
                + config.TRAIN.LRF  # noqa: E501
            )  # cosine
            xi = [0, num_warmup]
            for j, x in enumerate(optimizer.param_groups):
                x["lr"] = np.interp(
                    num_iter, xi, [config.TRAIN.WARMUP_BIASE_LR if j == 2 else 0.0, x["initial_lr"] * lf(epoch)]
                )  # noqa: E501
                if "momentum" in x:
                    x["momentum"] = np.interp(
                        num_iter, xi, [config.TRAIN.WARMUP_MOMENTUM, config.TRAIN.MOMENTUM]
                    )  # noqa: E501

        data_time.update(time.time() - start)
        if not config.DEBUG:
            input = input.to(device, non_blocking=True)
            assign_target = []
            for tgt in target:
                assign_target.append(tgt.to(device))
            target = assign_target
        with amp.autocast(enabled=device.type != "cpu"):
            outputs = model(input)
            total_loss = criterion(outputs, target, shapes)

        # compute gradient and do update step
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if config.vis_train_gt:
            if batch_i in [0, 1, 2]:
                image_ind = 1  # just the first image from three different batches
                img_inv_norm = inverse_normalize(
                    input[image_ind],
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                )
                img_test = cv2.cvtColor(
                    img_inv_norm.cpu().numpy().transpose(1, 2, 0) * 255, cv2.COLOR_BGR2RGB
                )  # noqa: E501
                da_gt_mask = target[1][image_ind].unsqueeze(0)
                _, da_gt_mask = torch.max(da_gt_mask, 1)
                da_gt_mask = da_gt_mask.int().squeeze().cpu().numpy()
                img_test1 = img_test.copy()
                _img_with_gt = show_seg_result(  # noqa: F841
                    img_test1,
                    da_gt_mask,
                    batch_i,
                    epoch,
                    save_dir,
                    is_gt=True,
                    config=config,
                    clearml_logger=clearml_logger,
                    prefix="train_",
                )

        # measure accuracy and record loss
        input_size = input.size(0)
        seg_da_loss.update(total_loss.item(), input_size)

        # measure elapsed time
        batch_time.update(time.time() - start)
        if batch_i % config.PRINT_FREQ == 0:
            msg = (
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t"
                "Speed {speed:.1f} samples/s\t"
                "Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t"
                "Loss {loss.val:.5f} ({loss.avg:.5f})".format(
                    epoch,
                    batch_i,
                    len(train_loader),
                    batch_time=batch_time,
                    speed=input.size(0) / batch_time.val,
                    data_time=data_time,
                    loss=seg_da_loss,
                )
            )
            logger.info(msg)

            writer = writer_dict["writer"]
            global_steps = writer_dict["train_global_steps"]
            writer.add_scalar("train_loss", seg_da_loss.val, global_steps)
            writer_dict["train_global_steps"] = global_steps + 1

        # BATCH END
        if config.DEBUG_N_BATCHES > 0 and batch_i == config.DEBUG_N_BATCHES:
            break

    if config.CLEARML_LOGGING:
        clearml_logger.current_logger().report_scalar(
            title="TRAIN_LOSS",
            series="TRAIN_LOSS",
            value=seg_da_loss.avg,
            iteration=epoch,
        )
        clearml_logger.current_logger().report_scalar(
            title="LOSSES", series="seg_da_LOSS", value=seg_da_loss.avg, iteration=epoch
        )


def validate(
    epoch: int,
    config: CfgNode,
    val_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: Callable,
    output_dir: str,
    device: Union[str, torch.device] = "cpu",
    clearml_logger: Optional[Any] = None,
    half: bool = False,
) -> Tuple[Tuple[float, float, float], float]:
    """
    Validate the model on the validation dataset.

    Args:
        epoch: Current epoch of training.
        config: Configuration object with parameters and hyperparameters.
        val_loader: DataLoader for the validation data.
        model: Model to validate.
        criterion: Loss function used for validation.
        output_dir: Directory where outputs will be saved.
        device: Device on which to perform the validation.
        clearml_logger: Logger for ClearML platform.
        half: Flag to indicate if half precision should be used for validation.

    Returns:
        A tuple containing the validation results in terms of accuracy and loss.
        - First element is a tuple of (average accuracy, average IoU, mean IoU).
        - Second element is the average loss.
    """
    # setting
    save_dir = output_dir + os.path.sep + "visualization"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    test_batch_size = config.TEST.BATCH_SIZE
    da_metric = SegmentationMetric(config.num_seg_class)

    losses = AverageMeter()
    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()
    da_mIoU_seg = AverageMeter()

    # switch to val mode
    model.eval()
    res_images = []

    for batch_i, (img, target, _paths, shapes) in tqdm(enumerate(val_loader), total=len(val_loader)):
        if not config.DEBUG:
            img = img.to(device, non_blocking=True)
            if half:
                img = img.half()
            assign_target = []
            for tgt in target:
                assign_target.append(tgt.to(device))
            target = assign_target
            nb, _, height, width = img.shape  # batch size, channel, height, width

        with torch.no_grad():
            pad_w, pad_h = shapes[0][1][1]
            pad_w = int(pad_w)
            pad_h = int(pad_h)

            da_seg_out = model(img)

            # driving area segment evaluation
            _, da_predict = torch.max(da_seg_out, 1)
            _, da_gt = torch.max(target[1], 1)
            da_predict = da_predict[:, pad_h : height - pad_h, pad_w : width - pad_w]
            da_gt = da_gt[:, pad_h : height - pad_h, pad_w : width - pad_w]

            da_metric.reset()
            da_metric.add_batch(da_predict.cpu(), da_gt.cpu())
            da_acc = da_metric.pixel_accuracy()
            da_IoU = da_metric.intersection_over_union()
            da_mIoU = da_metric.mean_intersection_over_union()

            da_acc_seg.update(da_acc, img.size(0))
            da_IoU_seg.update(da_IoU, img.size(0))
            da_mIoU_seg.update(da_mIoU, img.size(0))

            # compute loss
            total_loss = criterion(da_seg_out, target, shapes)
            losses.update(total_loss.item(), img.size(0))

            if config.TEST.PLOTS and not config.inference_visualization:
                if batch_i == 0 or batch_i == 1 or batch_i == 2:  # TO FIX
                    image_ind = 0  # just the first image from three diferent batches
                    img_inv_norm = inverse_normalize(
                        img[image_ind],
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    )
                    img_test = cv2.cvtColor(
                        img_inv_norm.cpu().numpy().transpose(1, 2, 0) * 255,
                        cv2.COLOR_BGR2RGB,
                    )

                    da_seg_mask = da_seg_out[image_ind].unsqueeze(0)
                    # if the first tensor (not road) max then 0, else 1 which is road
                    _, da_seg_mask = torch.max(da_seg_mask, 1)
                    da_gt_mask = target[1][image_ind].unsqueeze(0)
                    _, da_gt_mask = torch.max(da_gt_mask, 1)
                    da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
                    da_gt_mask = da_gt_mask.int().squeeze().cpu().numpy()

                    img_test1 = img_test.copy()
                    _img_with_segm = show_seg_result(  # noqa: F841
                        img_test,
                        da_seg_mask,
                        batch_i,
                        epoch,
                        save_dir,
                        config=config,
                        clearml_logger=clearml_logger,
                    )
                    _img_with_gt = show_seg_result(  # noqa: F841
                        img_test1,
                        da_gt_mask,
                        batch_i,
                        epoch,
                        save_dir,
                        is_gt=True,
                        config=config,
                        clearml_logger=clearml_logger,
                    )
            else:
                for image_ind in range(nb):  # for each image in batch
                    folder_to_save = Path(f"{save_dir}/inference_results/")
                    folder_to_save.mkdir(parents=True, exist_ok=True)

                    if config.save_gt:
                        folder_to_save_gt = Path(f"{save_dir}/gt/")
                        folder_to_save_gt.mkdir(parents=True, exist_ok=True)

                    if hasattr(val_loader.dataset, "data_path"):
                        filename = f"{val_loader.dataset.data_path((batch_i * test_batch_size) + image_ind).name[:-4]}.jpg"  # noqa: E501
                    else:
                        filename = f"{batch_i}_{image_ind}_det_pred.jpg"

                    img_inv_norm = inverse_normalize(
                        img[image_ind],
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    )
                    img_test = cv2.cvtColor(img_inv_norm.cpu().numpy().transpose(1, 2, 0) * 255, cv2.COLOR_BGR2RGB)

                    da_seg_mask = da_seg_out[image_ind].unsqueeze(0)
                    _, da_seg_mask = torch.max(da_seg_mask, 1)
                    da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()

                    img_test1 = img_test.copy()
                    img_with_predict = show_seg_result(
                        img_test1,
                        da_seg_mask,
                        batch_i,
                        epoch,
                        save_dir,
                        config=config,
                        clearml_logger=clearml_logger,
                    )
                    if config.vis_without_letterboxing:
                        img_with_predict = img_with_predict[pad_h : height - pad_h, pad_w : width - pad_w, :]
                    cv2.imwrite(f"{folder_to_save}/{filename}", img_with_predict)

                    if config.save_gt:
                        da_gt_mask = target[1][image_ind].unsqueeze(0)
                        _, da_gt_mask = torch.max(da_gt_mask, 1)
                        da_gt_mask = da_gt_mask.int().squeeze().cpu().numpy()
                        img_test2 = img_test.copy()
                        img_with_gt = show_seg_result(
                            img_test2,
                            da_gt_mask,
                            batch_i,
                            epoch,
                            save_dir,
                            is_gt=True,
                            config=config,
                            clearml_logger=clearml_logger,
                        )
                        cv2.imwrite(f"{folder_to_save_gt}/{filename}", img_with_gt)

                if config.save_video:
                    res_images.append(cv2.cvtColor(np.array(img_with_predict, dtype=np.uint8), cv2.COLOR_BGR2RGB))

        # BATCH END
        if config.DEBUG_N_BATCHES > 0 and batch_i == config.DEBUG_N_BATCHES:
            break

    if config.save_video:
        video_path = f"{save_dir}/segm_video_{val_loader.dataset.split}.mp4"
        write_video(res_images, video_path=video_path)

    model.float()  # for training
    da_segment_result = (da_acc_seg.avg, da_IoU_seg.avg, da_mIoU_seg.avg)

    return da_segment_result, losses.avg
