import numpy as np
from tqdm import tqdm

from .AutoDriveDataset3D import AutoDriveDataset3D

class SegmDataset3D(AutoDriveDataset3D):
    def __init__(self, cfg, is_train, inputsize, transform=None, data_path=None, split=None,
                 from_img=None, to_img=None):
        super().__init__(cfg, is_train, inputsize, transform, data_path, split, from_img,
                         to_img)
        self.db = self._get_db()
        self.cfg = cfg

    def _get_db(self):
        """
        get database from the annotation file

        Inputs:

        Returns:
        gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'image':, 'information':, ......}
        image: image path
        mask: path of the segmetation label
        label: [cls_id, center_x//256, center_y//256, w//256, h//256] 256=IMAGE_SIZE
        """
        print('building database...')
        gt_db = []
        for _ind, mask in tqdm(enumerate(self.mask_list)):
            mask_path = str(mask)
            points_path = mask_path.replace(self.mask_root.as_posix(), 
                                             self.points_root.as_posix())
            image_path  = mask_path.replace(self.mask_root.as_posix(), 
                                            self.img_root.as_posix()).replace(".npy", f".{self.cfg.DATASET.DATA_FORMAT}")
            gt = np.zeros((1, 5)) # zeros if we do not use this class!

            rec = [{
                'image': image_path,
                'label': gt,
                'mask': mask_path,
                'points': points_path,
            }]

            gt_db += rec
        print('database build finish')
        return gt_db
    
    def data_path(self, idx):
        return self.mask_list[idx]
