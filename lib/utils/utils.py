import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.optim as optim
from yacs.config import CfgNode


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Resets all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Updates the average meter with the new value and weight.

        Args:
            val (float): Value to update.
            n (int): Weight of the value.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def create_logger(cfg: CfgNode, cfg_path: str, phase: str = "train", rank: int = -1) -> Tuple:
    """
    Set up a logger for training or validation phase.

    Args:
        cfg: Configuration object with logging parameters.
        cfg_path: Path to the configuration file.
        phase: Current phase ('train' or 'val').
        rank: Rank of the process in distributed training to determine if logging is needed.

    Returns:
        A tuple containing the logger, output directory path, and tensorboard log
        directory path. If the rank is not in the main process, all three are None.
    """
    # set up logger dir
    dataset = cfg.DATASET.DATASET
    dataset = dataset.replace(":", "_")
    model = cfg.MODEL.NAME
    cfg_path = os.path.basename(cfg_path).split(".")[0]

    if rank in [-1, 0]:
        time_str = time.strftime("%Y-%m-%d-%H-%M")
        log_file = "{}_{}_{}.log".format(cfg_path, time_str, phase)
        # set up tensorboard_log_dir
        tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / (cfg_path + "_" + time_str)
        final_output_dir = tensorboard_log_dir
        if not tensorboard_log_dir.exists():
            print("=> creating {}".format(tensorboard_log_dir))
            tensorboard_log_dir.mkdir(parents=True)

        final_log_file = tensorboard_log_dir / log_file
        head = "%(asctime)-15s %(message)s"
        logging.basicConfig(filename=str(final_log_file), format=head)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        logging.getLogger("").addHandler(console)

        return logger, str(final_output_dir), str(tensorboard_log_dir)
    else:
        return None, None, None


def get_optimizer(cfg: Any, model: torch.nn.Module) -> Optional[torch.optim.Optimizer]:
    """
    Create an optimizer for the model based on the configuration.

    Args:
        cfg: Configuration object with optimizer settings.
        model: The model for which to create the optimizer.

    Returns:
        An instance of torch.optim.Optimizer according to the specified configuration.
        Returns None if the specified optimizer is not supported.
    """
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR0,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV,
        )
    elif cfg.TRAIN.OPTIMIZER == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR0,
            betas=(cfg.TRAIN.MOMENTUM, 0.999),
        )
    return optimizer


def save_checkpoint(
    epoch: int,
    name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: str,
    filename: str,
    is_best: bool = False,
) -> None:
    """
    Save the model checkpoint.

    Args:
        epoch: Current epoch of training.
        name: Name of the model.
        model: The model to be saved.
        optimizer: The optimizer with current state to be saved.
        output_dir: Directory where to save the checkpoint.
        filename: Name of the checkpoint file.
        is_best: If True, also saves the checkpoint as the best model so far.
    """
    model_state = model.state_dict()
    checkpoint = {
        "epoch": epoch,
        "model": name,
        "state_dict": model_state,
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, os.path.join(output_dir, filename))
    if is_best and "state_dict" in checkpoint:
        torch.save(checkpoint["best_state_dict"], os.path.join(output_dir, "model_best.pth"))


def xyxy2xywh(x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert format from top-left and bottom-right points (x1, y1, x2, y2)
    to top-left point and width, height (x, y, w, h).

    Args:
        x: Array or tensor with shape (N, 4).

    Returns:
        Converted bounding box array or tensor with the same shape as input.
    """
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h]
    # where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height

    return y


def time_synchronized() -> float:
    """
    Return the current time in seconds, synchronized across different devices
    if using CUDA.

    Returns:
        Current time in seconds as a float.
    """
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def inverse_normalize(tensor: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """
    Apply inverse normalization to the given tensor using the specified mean and std.

    Args:
        tensor: The tensor to be inverse normalized.
        mean: The mean used for normalization.
        std: The standard deviation used for normalization.

    Returns:
        The inverse normalized tensor.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


@contextmanager
def open_video(
    video_path: Union[str, Path], mode: str = "r", *args: Any
) -> Iterator[Union[cv2.VideoCapture, cv2.VideoWriter]]:
    """Context manager to work with cv2 videos
        Mimics python's standard `open` function

    Args:
        video_path: path to video to open
        mode: either 'r' for read or 'w' write
        args: additional arguments passed to Capture or Writer
            according to OpenCV documentation
    Returns:
        cv2.VideoCapture or cv2.VideoWriter depending on mode

    Example of writing:
        open_video(
            out_path,
            'w',
            cv2.VideoWriter_fourcc(*'XVID'), # fourcc
            15, # fps
            (width, height), # frame size
        )
    """
    video_path = Path(video_path)
    if mode == "r":
        video = cv2.VideoCapture(video_path.as_posix(), *args)
    elif mode == "w":
        video = cv2.VideoWriter(video_path.as_posix(), *args)
    else:
        raise ValueError(f'Incorrect open mode "{mode}"; "r" or "w" expected!')
    if not video.isOpened():
        raise ValueError(f"Video {video_path} is not opened!")
    try:
        yield video
    finally:
        video.release()


def write_video(
    images: List[np.ndarray],
    video_path: Union[str, Path],
    codec_code: str = "XVID",
    fps: int = 2,
    is_color: bool = True,
) -> None:
    """
    Write a sequence of images to a video file.

    Args:
        images: A list of images, each as an array in RGB format.
        video_path: The file path where the video will be saved.
        codec_code: Codec used to compress the frames, e.g., 'XVID' for .avi format.
        fps: Frames per second in the output video file.
        is_color: Whether the images are in color.
    """

    fourcc = cv2.VideoWriter_fourcc(*codec_code)
    height, width, channels = images[0].shape
    with open_video(video_path, "w", fourcc, fps, (width, height), is_color) as capture:
        for frame in images:
            if is_color:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            capture.write(frame)
