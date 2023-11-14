import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from ..utils import augment_hsv, letterbox, random_perspective, xyxy2xywh


class AutoDriveDataset3D(Dataset):
    """
    A general Dataset for some common function
    """

    def __init__(
        self,
        cfg,
        is_train,
        inputsize=640,
        transform=None,
        data_path=None,
        split=None,
        from_img=None,
        to_img=None,
    ):
        """
        initial all the characteristic

        Inputs:
        -cfg: configurations
        -is_train(bool): whether train set or not
        -transform: ToTensor and Normalize

        Returns:
        None
        """
        self.cfg = cfg
        self.is_train = is_train
        self.inputsize = inputsize
        self.transform = transform
        self.split = split
        self.Tensor = transforms.ToTensor()

        if data_path:
            img_root = Path(f"{data_path}/images")
            mask_root = Path(f"{data_path}/seg_points")
            points_root = Path(f"{data_path}/seg_points_total")
        else:
            img_root = Path(cfg.DATASET.DATAROOT)
            mask_root = Path(cfg.DATASET.MASKROOT)

        self.img_root = img_root / split
        self.mask_root = mask_root / split
        self.points_root = points_root / split

        self.img_list = sorted(list(self.img_root.iterdir()))
        self.mask_list = sorted(list(self.mask_root.iterdir()))
        self.points_list = sorted(list(self.points_root.iterdir()))
        if from_img is not None:
            self.img_list = self.img_list[from_img:to_img]
            self.mask_list = self.mask_list[from_img:to_img]
            self.points_list = self.points_list[from_img:to_img]

        self.db = []

        self.data_format = cfg.DATASET.DATA_FORMAT

    def _get_db(self):
        """ """
        raise NotImplementedError

    def __len__(self):
        """
        number of objects in the dataset
        """
        return len(self.db)

    def __getitem__(self, idx):
        """
        Get input and groud-truth from database & add data augmentation on input

        Inputs:
        -idx: the index of image in self.db(database)(list)
        self.db(list) [a,b,c,...]
        a: (dictionary){'image':, 'information':}

        Returns:
        -image: transformed image, first passed the data augmentation in
            __getitem__ function(type:numpy), then apply self.transform
        -target: ground truth(det_gt,seg_gt)

        function maybe useful
        cv2.imread
        cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        cv2.warpAffine
        """
        data = self.db[idx]
        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        seg_label = np.zeros(img.shape[:2])
        total_points_label = np.zeros(img.shape[:2])

        resized_shape = self.inputsize
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)
        h0, w0 = img.shape[:2]  # orig hw
        r = resized_shape / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            seg_label = cv2.resize(
                seg_label, (int(w0 * r), int(h0 * r)), interpolation=interp
            )
            total_points_label = cv2.resize(
                total_points_label, (int(w0 * r), int(h0 * r)), interpolation=interp
            )
        h, w = img.shape[:2]

        # gt points
        points_r = 1 / r
        segm_points = np.load(Path(data["mask"]))
        segm_points = segm_points // points_r
        if len(segm_points.shape) != 2:
            segm_points = np.empty((0, 2))

        segm_points[:, 0] = np.clip(segm_points[:, 0], 0, int(h0 * r) - 1)
        segm_points[:, 1] = np.clip(segm_points[:, 1], 0, int(w0 * r) - 1)
        x = [int(x) for y, x in segm_points]
        y = [int(y) for y, x in segm_points]

        if self.cfg.DATASET.FILL_BETWEEN_POINTS:
            y_border = int(h0 * r) - 1
            x_add1 = [int(x) for y, x in segm_points if y > y_border / 2]
            y_add1 = [
                np.clip(int(y + 1), 0, y_border)
                for y, x in segm_points
                if y > y_border / 2
            ]
            x_add2 = [int(x) for y, x in segm_points if y > y_border / 2]
            y_add2 = [
                np.clip(int(y - 1), 0, y_border)
                for y, x in segm_points
                if y > y_border / 2
            ]

            seg_label[y_add1, x_add1] = 1
            seg_label[y_add2, x_add2] = 1

        seg_label[y, x] = 1
        seg_label = np.array(seg_label, dtype=np.uint8)

        # total points for mask
        total_points = np.load(Path(data["points"]))
        total_points = total_points // points_r  # TO FIX spcific r
        total_points[:, 0] = np.clip(total_points[:, 0], 0, int(h0 * r) - 1)
        total_points[:, 1] = np.clip(total_points[:, 1], 0, int(w0 * r) - 1)
        x = [int(x) for x, y in total_points]
        y = [int(y) for x, y in total_points]
        total_points_label[x, y] = 1
        total_points_label = np.array(total_points_label, dtype=np.uint8)

        # we add noise to original image only
        noise = (np.random.rand(*total_points_label.shape) > 0.95).astype(np.uint8)
        # noise only for an upper half of the image
        no_noise = np.zeros(
            (total_points_label.shape[0] // 2, total_points_label.shape[1])
        )
        noise[total_points_label.shape[0] // 2 :, :] = no_noise

        total_points_label = np.clip(total_points_label + noise, 0, 1)

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
            labels[:, 1] = (
                ratio[0] * w * (det_label[:, 1] - det_label[:, 3] / 2) + pad[0]
            )  # pad width
            labels[:, 2] = (
                ratio[1] * h * (det_label[:, 2] - det_label[:, 4] / 2) + pad[1]
            )  # pad height
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

        # convert to 0-255 to feed to self.Tensor
        _, seg1 = cv2.threshold(seg_label, 0, 255, cv2.THRESH_BINARY)  # gt road
        _, seg2 = cv2.threshold(seg_label, 0, 255, cv2.THRESH_BINARY_INV)  # inverse

        seg1 = self.Tensor(seg1.copy())
        seg2 = self.Tensor(seg2.copy())

        # dilation + erosion
        if self.cfg.DATASET.WAYMO_DILATION:
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.dilate(np.array(seg1[0]), kernel, iterations=7)
            processed = cv2.erode(processed, kernel, iterations=3)
            seg1 = torch.Tensor(processed).unsqueeze(dim=0)
            seg2 = torch.Tensor((processed - 1.0) * -1.0).unsqueeze(dim=0)

        seg_label = torch.stack((seg2[0], seg1[0]), 0)

        # we add noise to image with borders
        # noise = (np.random.rand(*total_points_label.shape) > 0.97).astype(np.uint8)
        # total_points_label = np.clip(total_points_label + noise, 0, 1)

        # convert to 0-255 to feed to self.Tensor
        _, points1 = cv2.threshold(
            total_points_label, 0, 255, cv2.THRESH_BINARY
        )  # mask of points
        _, points2 = cv2.threshold(
            total_points_label, 0, 255, cv2.THRESH_BINARY
        )  # the same mask

        points1 = self.Tensor(points1.copy())
        points2 = self.Tensor(points2.copy())

        total_points_label = torch.stack((points2[0], points1[0]), 0)

        target = [labels_out, seg_label, total_points_label]
        img = self.transform(img)

        return img, target, data["image"], shapes

    @staticmethod
    def collate_fn(batch):
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
            [
                torch.cat(label_det, 0),
                torch.stack(label_seg, 0),
                torch.stack(label_points, 0),
            ],
            paths,
            shapes,
        )
