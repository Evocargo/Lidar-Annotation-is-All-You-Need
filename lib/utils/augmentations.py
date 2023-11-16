import math
import random
from typing import Tuple

import cv2
import numpy as np


def augment_hsv(img: np.ndarray, hgain: float = 0.5, sgain: float = 0.5, vgain: float = 0.5) -> None:
    """
    Apply Hue, Saturation, and Value (HSV) color space augmentation.

    Args:
        img: Image (numpy array) to be augmented.
        hgain: Hue gain for the augmentation.
        sgain: Saturation gain for the augmentation.
        vgain: Value gain for the augmentation.
    """
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def random_perspective(
    combination: Tuple[np.ndarray, np.ndarray, np.ndarray],
    targets: np.ndarray = (),
    degrees: float = 10,
    translate: float = 0.1,
    scale: float = 0.1,
    shear: float = 10,
    perspective: float = 0.0,
    border: Tuple[int, int] = (0, 0),
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """
    Perform a random perspective transformation on the image and its corresponding targets.

    Args:
        combination: A tuple containing the image, grayscale image, and line image.
        targets: Ground truth target values.
        degrees: Range of degrees for random rotations.
        translate: Range of translation.
        scale: Scale range for zooming.
        shear: Shear angle range.
        perspective: Perspective range.
        border: Tuple of pixel values to use for border.

    Returns:
        Tuple of the transformed combination (image, grayscale, line) and updated targets.
    """
    img, gray, line = combination
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - (scale * 0.5), 1 + scale)  # CHANGED this to get more zoomed images TO FIX
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            gray = cv2.warpPerspective(gray, M, dsize=(width, height), borderValue=0)
            line = cv2.warpPerspective(line, M, dsize=(width, height), borderValue=0)
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            # interpolation flag here to more precise transform for points on image
            gray = cv2.warpAffine(
                gray, M[:2], dsize=(width, height), borderValue=0, flags=cv2.INTER_NEAREST
            )  # or INTER_LANCZOS4
            line = cv2.warpAffine(line, M[:2], dsize=(width, height), borderValue=0)

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = _box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, 1:5] = xy[i]

    combination = (img, gray, line)
    return combination, targets


def letterbox(
    combination: Tuple[np.ndarray, np.ndarray, np.ndarray],
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scale_fill: bool = False,
    scaleup: bool = True,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[float, float], Tuple[int, int]]:
    """
    Resize image in a way that keeps the aspect ratio and adds padding.
    https://zhuanlan.zhihu.com/p/172121380

    Args:
        combination: A tuple containing the image, grayscale image, and line image.
        new_shape: New image shape.
        color: Border color.
        auto: Adjusts to the minimum rectangle if True.
        scale_fill: Stretches the image and fills the box if True.
        scaleup: Allows the image to scale up if True.

    Returns:
        Tuple of the transformed combination (image, grayscale, line), the resize ratio,
        and padding dimensions.
    """

    img, gray, line = combination
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        gray = cv2.resize(gray, new_unpad, interpolation=cv2.INTER_LINEAR)
        line = cv2.resize(line, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    gray = cv2.copyMakeBorder(gray, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)  # add border
    line = cv2.copyMakeBorder(line, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)  # add border

    combination = (img, gray, line)
    return combination, ratio, (dw, dh)


def letterbox_for_img(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scale_fill: bool = False,
    scaleup: bool = True,
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """
    Resize a single image in a way that keeps the aspect ratio and adds padding.

    Args:
        img: Image to be resized and padded.
        new_shape: New image shape.
        color: Border color.
        auto: Adjusts to the minimum rectangle if True.
        scale_fill: Stretches the image and fills the box if True.
        scaleup: Allows the image to scale up if True.

    Returns:
        Tuple of the transformed image, the resize ratio, and padding dimensions.
    """
    # Resize image to a 32-pixel-multiple rectangle
    # https://github.com/ultralytics/yolov3/issues/232

    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding

    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def _box_candidates(
    box1: np.ndarray, box2: np.ndarray, wh_thr: int = 2, ar_thr: int = 20, area_thr: float = 0.1
) -> np.ndarray:
    """
    Compute candidate boxes that meet the specified thresholds.

    Args:
        box1: Original boxes before augmentation.
        box2: Boxes after augmentation.
        wh_thr: Width and height threshold.
        ar_thr: Aspect ratio threshold.
        area_thr: Area ratio threshold.

    Returns:
        Numpy array of candidate boxes that meet the specified thresholds.
    """
    # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment,
    # box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates
