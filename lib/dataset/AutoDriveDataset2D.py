import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from yacs.config import CfgNode

from ..utils import augment_hsv, letterbox, random_perspective, xyxy2xywh


class AutoDriveDataset2D(Dataset):
    """A base Dataset class handling 2D segmentation masks."""

    def __init__(
        self,
        cfg: CfgNode,
        is_train: bool,
        inputsize: Union[int, List[int]] = 640,
        transform: Optional[Callable] = None,
        data_path: Optional[str] = None,
        split: Optional[str] = None,
        from_img: Optional[int] = None,
        to_img: Optional[int] = None,
    ):
        """
        Initialize the dataset with given configuration and path.

        Args:
            cfg: Configuration object with dataset parameters.
            is_train (bool): Whether to set the dataset mode as training.
            inputsize (int, list): The input size for images. Defaults to 640.
            transform (callable, optional): Transformations to apply to the images.
            data_path (str, optional): Custom path to dataset images and masks.
            split (str, optional): The dataset split, e.g., 'train', 'val'.
            from_img (int, optional): Index to slice the dataset from.
            to_img (int, optional): Index to slice the dataset to.
        """
        self.cfg = cfg
        self.is_train = is_train
        self.inputsize = inputsize
        self.transform = transform
        self.split = split
        self.Tensor = transforms.ToTensor()

        if data_path:
            img_root = Path(f"{data_path}/images")
            mask_root = Path(f"{data_path}/seg_masks")
        else:
            img_root = Path(cfg.DATASET.DATAROOT)
            mask_root = Path(cfg.DATASET.MASKROOT)

        self.img_root = img_root / split
        self.mask_root = mask_root / split

        self.img_list = sorted(list(self.img_root.iterdir()))
        self.mask_list = sorted(list(self.mask_root.iterdir()))
        if from_img is not None:
            self.img_list = self.img_list[from_img:to_img]
            self.mask_list = self.mask_list[from_img:to_img]

        self.db = []

        self.data_format = cfg.DATASET.DATA_FORMAT

    def _get_db(self):
        """Placeholder method for fetching the database.

        Raises:
            NotImplementedError: If the method is not overridden in a subclass.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of objects in the dataset."""
        return len(self.db)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Get item from dataset at a specific index.

        Args:
            idx (int): Index of the data to fetch from the dataset.

        Returns:
            A tuple containing the transformed image, the targets (ground truth),
            the path of the image, and the shapes for resizing during evaluation.
        """
        data = self.db[idx]
        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        seg_label = cv2.imread(data["mask"])
        total_points_label = np.zeros(img.shape[:2])  # empty

        resized_shape = self.inputsize
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)
        h0, w0 = img.shape[:2]  # orig hw
        r = resized_shape / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            seg_label = cv2.resize(seg_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
            total_points_label = cv2.resize(total_points_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w = img.shape[:2]

        (img, seg_label, total_points_label), ratio, pad = letterbox(
            (img, seg_label, total_points_label),
            resized_shape,
            auto=self.cfg.DATASET.AUTO_SHAPE,
            scaleup=self.is_train,
        )
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        det_label = data["label"]
        labels = []

        if det_label.size > 0:
            # Normalized xywh to pixel xyxy format
            labels = det_label.copy()
            labels[:, 1] = ratio[0] * w * (det_label[:, 1] - det_label[:, 3] / 2) + pad[0]  # pad width
            labels[:, 2] = ratio[1] * h * (det_label[:, 2] - det_label[:, 4] / 2) + pad[1]  # pad height
            labels[:, 3] = ratio[0] * w * (det_label[:, 1] + det_label[:, 3] / 2) + pad[0]
            labels[:, 4] = ratio[1] * h * (det_label[:, 2] + det_label[:, 4] / 2) + pad[1]

        if self.is_train:
            combination = (img, seg_label, total_points_label)
            (img, seg_label, total_points_label), labels = random_perspective(
                combination=combination,
                targets=labels,
                degrees=self.cfg.DATASET.ROT_FACTOR,
                translate=self.cfg.DATASET.TRANSLATE,
                scale=self.cfg.DATASET.SCALE_FACTOR,
                shear=self.cfg.DATASET.SHEAR,
            )

            augment_hsv(
                img,
                hgain=self.cfg.DATASET.HSV_H,
                sgain=self.cfg.DATASET.HSV_S,
                vgain=self.cfg.DATASET.HSV_V,
            )

            if len(labels):
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

                # Normalize coordinates 0 - 1
                labels[:, [2, 4]] /= img.shape[0]  # height
                labels[:, [1, 3]] /= img.shape[1]  # width

            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                seg_label = np.fliplr(seg_label)
                total_points_label = np.fliplr(total_points_label)
                if len(labels):
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                seg_label = np.filpud(seg_label)
                total_points_label = np.filpud(total_points_label)
                if len(labels):
                    labels[:, 2] = 1 - labels[:, 2]

        else:
            if len(labels):
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

                # Normalize coordinates 0 - 1
                labels[:, [2, 4]] /= img.shape[0]  # height
                labels[:, [1, 3]] /= img.shape[1]  # width

        labels_out = torch.zeros((len(labels), 6))
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = np.ascontiguousarray(img)

        # dummy masks of ones
        points1 = np.ones(seg_label.shape)
        points2 = np.ones(seg_label.shape)

        # convert to 0-255 to feed to self.Tensor
        _, seg1 = cv2.threshold(seg_label, 0, 255, cv2.THRESH_BINARY)  # gt road
        _, seg2 = cv2.threshold(seg_label, 0, 255, cv2.THRESH_BINARY_INV)  # inverse

        seg1 = self.Tensor(seg1.copy())
        seg2 = self.Tensor(seg2.copy())

        seg_label = torch.stack((seg2[0], seg1[0]), 0)

        points1 = self.Tensor(points1.copy())
        points2 = self.Tensor(points2.copy())

        total_points_label = torch.stack((points2[0], points1[0]), 0)

        target = [labels_out, seg_label, total_points_label]
        img = self.transform(img)

        return img, target, data["image"], shapes

    @staticmethod
    def collate_fn(batch: List) -> Tuple:
        """
        Custom collate function to combine multiple data samples into a batch.

        Args:
            batch (list): List of tuples containing images, labels, paths, and shapes.

        Returns:
            A tuple containing batches of images, labels
            (detection, segmentation, points), paths of the images,
            and shapes for resizing.
        """

        img, label, paths, shapes = zip(*batch)
        label_det, label_seg, label_points = [], [], []
        for i, l in enumerate(label):
            l_det, l_seg, l_points = l
            l_det[:, 0] = i  # add target image index for build_targets()
            label_det.append(l_det)
            label_seg.append(l_seg)
            label_points.append(l_points)

        return (
            torch.stack(img, 0),
            [torch.cat(label_det, 0), torch.stack(label_seg, 0), torch.stack(label_points, 0)],
            paths,
            shapes,
        )
