import numpy as np


class SegmentationMetric(object):
    """
    img_label [batch_size, height(144), width(256)]
    confusion_matrix [[0(TN),1(FP)],
                     [2(FN),3(TP)]]
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((self.num_classes,) * 2)

    def pixel_accuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def line_accuracy(self):
        acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + 1e-12)
        return acc[1]

    def class_pixel_accuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        class_acc = np.diag(self.confusion_matrix) / (
            self.confusion_matrix.sum(axis=0) + 1e-12
        )
        return class_acc

    def mean_pixel_accuracy(self):
        class_acc = self.class_pixel_accuracy()
        mean_acc = np.nanmean(class_acc)
        return mean_acc

    def mean_intersection_over_union(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusion_matrix)
        union = (
            np.sum(self.confusion_matrix, axis=1)
            + np.sum(self.confusion_matrix, axis=0)
            - np.diag(self.confusion_matrix)
        )
        iou = intersection / union
        iou[np.isnan(iou)] = 0
        miou = np.nanmean(iou)
        return miou

    def intersection_over_union(self):
        intersection = np.diag(self.confusion_matrix)
        union = (
            np.sum(self.confusion_matrix, axis=1)
            + np.sum(self.confusion_matrix, axis=0)
            - np.diag(self.confusion_matrix)
        )
        iou = intersection / union
        iou[np.isnan(iou)] = 0
        return iou[1]

    def gen_confusion_matrix(self, img_predict, img_label):
        # remove classes from unlabeled pixels in gt image and predict

        mask = (img_label >= 0) & (img_label < self.num_classes)
        label = self.num_classes * img_label[mask] + img_predict[mask]
        count = np.bincount(label, minlength=self.num_classes ** 2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        return confusion_matrix

    def frequency_weighted_intersection_over_union(self):
        # FWIOU = [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1)
            + np.sum(self.confusion_matrix, axis=0)
            - np.diag(self.confusion_matrix)
        )
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def add_batch(self, img_predict, img_label):
        assert img_predict.shape == img_label.shape
        self.confusion_matrix += self.gen_confusion_matrix(img_predict, img_label)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
