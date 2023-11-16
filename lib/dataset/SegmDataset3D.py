from typing import Callable, List, Optional, Union

import numpy as np
from tqdm import tqdm
from yacs.config import CfgNode

from .AutoDriveDataset3D import AutoDriveDataset3D


class SegmDataset3D(AutoDriveDataset3D):
    """Dataset for semantic segmentation task using 3D reprojected points as masks."""

    def __init__(
        self,
        cfg: CfgNode,
        is_train: bool,
        inputsize: Union[int, List[int]],
        transform: Optional[Callable] = None,
        data_path: Optional[str] = None,
        split: Optional[str] = None,
        from_img: Optional[int] = None,
        to_img: Optional[int] = None,
    ):
        super().__init__(cfg, is_train, inputsize, transform, data_path, split, from_img, to_img)
        self.db = self._get_db()
        self.cfg = cfg

    def _get_db(self) -> List[dict]:
        """
        Construct the database from the annotation file.

        The database is built by associating each segmentation mask with its corresponding
        image and an array of zeros representing the absence of classification labels
        in this context.

        Returns:
            gt_db (list of dicts): Database containing image and segmentation mask paths,
            and placeholder for classification labels. Example structure:
                [{'image': image_path,
                  'label': gt,
                  'mask': mask_path,
                  'points': points_path}, ...]
        """
        print("building database...")
        gt_db = []
        for _ind, mask in tqdm(enumerate(self.mask_list)):
            mask_path = str(mask)
            points_path = mask_path.replace(self.mask_root.as_posix(), self.points_root.as_posix())
            image_path = mask_path.replace(self.mask_root.as_posix(), self.img_root.as_posix()).replace(
                ".npy", f".{self.cfg.DATASET.DATA_FORMAT}"
            )
            gt = np.zeros((1, 5))  # zeros if we do not use this class!

            rec = [
                {
                    "image": image_path,
                    "label": gt,
                    "mask": mask_path,
                    "points": points_path,
                }
            ]

            gt_db += rec
        print("database build finish")
        return gt_db

    def data_path(self, idx: int) -> str:
        """
        Get the data path for a given index.

        Args:
            idx (int): The index of the data.

        Returns:
            The file path of the mask associated with the given index.
        """
        return self.mask_list[idx]
