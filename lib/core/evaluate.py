import numpy as np


class SegmentationMetric:
    """
    Class to calculate segmentation metrics including pixel accuracy, mean pixel accuracy,
    mean Intersection over Union (IoU), and frequency-weighted IoU.
    """

    def __init__(self, num_classes: int):
        """
        Initializes the SegmentationMetric class with the number of classes.

        Args:
            num_classes (int): The number of classes in the segmentation task.
        """
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((self.num_classes,) * 2)

    def pixel_accuracy(self) -> float:
        """
        Calculates the pixel accuracy across all classes.

        Returns:
            float: Overall pixel accuracy.
        """
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def class_pixel_accuracy(self) -> np.ndarray:
        """
        Calculates the pixel accuracy for each class individually.

        Returns:
            np.ndarray: Array containing pixel accuracy for each class.
        """
        class_acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=0) + 1e-12)
        return class_acc

    def mean_pixel_accuracy(self) -> float:
        """
        Calculate the mean pixel accuracy across all classes.

        Returns:
            float: Mean pixel accuracy.
        """
        class_acc = self.class_pixel_accuracy()
        mean_acc = np.nanmean(class_acc)
        return mean_acc

    def mean_intersection_over_union(self) -> float:
        """
        Calculates the mean Intersection over Union (IoU) across all classes.

        Returns:
            float: Mean IoU.
        """
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

    def intersection_over_union(self) -> float:
        """
        Calculates the Intersection over Union (IoU) for the second class.

        Returns:
            float: IoU for the second class.
        """
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
        """
        Generates the confusion matrix for a batch of predictions and labels.

        Args:
            img_predict (np.ndarray): Array of predicted labels.
            img_label (np.ndarray): Array of ground truth labels.

        Returns:
            np.ndarray: The confusion matrix for the current batch.
        """
        # remove classes from unlabeled pixels in gt image and predict

        mask = (img_label >= 0) & (img_label < self.num_classes)
        label = self.num_classes * img_label[mask] + img_predict[mask]
        count = np.bincount(label, minlength=self.num_classes ** 2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        return confusion_matrix

    def frequency_weighted_intersection_over_union(self) -> float:
        """
        Calculates the frequency-weighted Intersection over Union (FWIoU).

        Returns:
            float: Frequency-weighted IoU.
        """

        # FWIOU = [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1)
            + np.sum(self.confusion_matrix, axis=0)
            - np.diag(self.confusion_matrix)
        )
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def add_batch(self, img_predict: np.ndarray, img_label: np.ndarray) -> None:
        """
        Adds a new batch of predictions and labels to the confusion matrix.

        Args:
             img_predict (np.ndarray): Array of predicted labels.
             img_label (np.ndarray): Array of ground truth labels.
        """
        assert img_predict.shape == img_label.shape
        self.confusion_matrix += self.gen_confusion_matrix(img_predict, img_label)

    def reset(self) -> None:
        """
        Resets the confusion matrix to zeros.
        """
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
