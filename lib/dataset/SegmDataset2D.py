import numpy as np
from tqdm import tqdm
from .AutoDriveDataset2D import AutoDriveDataset2D


class SegmDataset2D(AutoDriveDataset2D):
    def __init__(self, cfg, is_train, inputsize, transform=None, data_path=None, split=None,
                 from_img=None, to_img=None):
        super().__init__(cfg, is_train, inputsize, transform, data_path, split, from_img, to_img)
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
        for _ind, img in tqdm(enumerate(self.img_list)):
            image_path = str(img)
            mask_path  = image_path.replace(self.img_root.as_posix(), 
                                            self.mask_root.as_posix()).replace(".jpg", ".png")
            gt = np.zeros((1, 5)) # zeros if we do not use this class!

            rec = [{
                'image': image_path,
                'label': gt,
                'mask': mask_path,
            }]

            gt_db += rec
        print('database build finish')
        return gt_db

    def data_path(self, idx):
        return self.img_list[idx]