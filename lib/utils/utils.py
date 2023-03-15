import os
import logging
import time
from pathlib import Path
import cv2
import torch
import torch.optim as optim
import numpy as np
from contextlib import contextmanager

def create_logger(cfg, cfg_path, phase='train', rank=-1):
    # set up logger dir
    dataset = cfg.DATASET.DATASET
    dataset = dataset.replace(':', '_')
    model = cfg.MODEL.NAME
    cfg_path = os.path.basename(cfg_path).split('.')[0]

    if rank in [-1, 0]:
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        log_file = '{}_{}_{}.log'.format(cfg_path, time_str, phase)
        # set up tensorboard_log_dir
        tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
                                  (cfg_path + '_' + time_str)
        final_output_dir = tensorboard_log_dir
        if not tensorboard_log_dir.exists():
            print('=> creating {}'.format(tensorboard_log_dir))
            tensorboard_log_dir.mkdir(parents=True)

        final_log_file = tensorboard_log_dir / log_file
        head = '%(asctime)-15s %(message)s'
        logging.basicConfig(filename=str(final_log_file),
                            format=head)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)

        return logger, str(final_output_dir), str(tensorboard_log_dir)
    else:
        return None, None, None

def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR0,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR0,
            betas=(cfg.TRAIN.MOMENTUM, 0.999)
        )
    return optimizer

def save_checkpoint(epoch, name, model, optimizer, output_dir, filename, is_best=False):
    model_state = model.state_dict()
    checkpoint = {
            'epoch': epoch,
            'model': name,
            'state_dict': model_state,
            'optimizer': optimizer.state_dict(),
        }
    torch.save(checkpoint, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in checkpoint:
        torch.save(checkpoint['best_state_dict'],
                   os.path.join(output_dir, 'model_best.pth'))

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

@contextmanager
def open_video(video_path, mode="r", *args):
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
    images,
    video_path,
    codec_code: str = "XVID",
    fps: int = 2,
    is_color=True,
):
    """

    Args:
        images: List of RGB or binary images.
        video_path: The name of the file to save the video to.
        codec_code: FourCC - a 4-byte code used to specify the video codec.
        fps: Framerate of the created video stream.
        is_color: RGB images or not.

    """
    fourcc = cv2.VideoWriter_fourcc(*codec_code)
    height, width, channels = images[0].shape
    with open_video(video_path, "w", fourcc, fps, (width, height), is_color) as capture:
        for frame in images:
            if is_color:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            capture.write(frame)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0