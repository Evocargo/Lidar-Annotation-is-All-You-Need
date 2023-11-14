from typing import Any, Optional

import cv2
import numpy as np
from yacs.config import CfgNode


def show_seg_result(
    img: np.ndarray,
    result: np.ndarray,
    index: int,
    epoch: int,
    save_dir: Optional[str] = None,
    palette: Optional[np.ndarray] = None,
    is_demo: bool = False,
    is_gt: bool = False,
    config: Optional[CfgNode] = None,
    clearml_logger: Optional[Any] = None,
    prefix: str = "",
) -> np.ndarray:
    """
    Overlay segmentation results on the image and save or log the visualization.

    Args:
        img: The original image as a numpy array.
        result: Segmentation results as a numpy array.
        index: Index of the batch or image.
        epoch: Current epoch number for logging purposes.
        save_dir: Directory path where the image will be saved.
        palette: Color palette for segmentation classes.
        is_demo: Flag indicating if this is a demo (changes visualization style).
        is_gt: Flag indicating if the result is ground truth segmentation.
        config: Configuration object containing parameters for saving and logging.
        clearml_logger: Logger for the ClearML platform.
        prefix: Prefix string for the saved or logged image file name.

    Returns:
        The image with segmentation results overlaid as a numpy array.
    """

    if palette is None:
        palette = np.random.randint(0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)
    assert palette.shape[0] == 3  # len(classes)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2

    if not is_demo:
        color_seg = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[result == label, :] = color
    else:
        color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)
        color_area[result[0] == 1] = [0, 255, 0]
        color_area[result[1] == 1] = [255, 0, 0]
        color_seg = color_area

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    color_mask = np.mean(color_seg, 2)

    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5

    if not is_demo:
        if not is_gt:
            if config.TRAIN.SAVE_LOCALLY_PER_BATCH:
                cv2.imwrite(save_dir + f"/{prefix}batch_{epoch}_{index}_da_segresult.jpg", img)
            if config.CLEARML_LOGGING:
                clearml_logger.current_logger().report_image(
                    "image",
                    f"{prefix}da_segresult{index}",
                    iteration=epoch,
                    image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                )
        else:
            if config.TRAIN.SAVE_LOCALLY_PER_BATCH:
                cv2.imwrite(save_dir + f"/{prefix}batch_{epoch}_{index}_da_seg_gt.jpg", img)
            if config.CLEARML_LOGGING:
                clearml_logger.current_logger().report_image(
                    "image",
                    f"{prefix}da_seg_gt{index}",
                    iteration=epoch,
                    image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                )

    return img
