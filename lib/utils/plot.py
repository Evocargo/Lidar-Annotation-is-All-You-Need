import cv2
import numpy as np


def show_seg_result(
    img,
    result,
    index,
    epoch,
    save_dir=None,
    palette=None,
    is_demo=False,
    is_gt=False,
    config=None,
    clearml_logger=None,
    prefix="",
):

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
                cv2.imwrite(
                    save_dir + f"/{prefix}batch_{epoch}_{index}_da_segresult.jpg", img
                )
            if config.CLEARML_LOGGING:
                clearml_logger.current_logger().report_image(
                    "image",
                    f"{prefix}da_segresult{index}",
                    iteration=epoch,
                    image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                )
        else:
            if config.TRAIN.SAVE_LOCALLY_PER_BATCH:
                cv2.imwrite(
                    save_dir + f"/{prefix}batch_{epoch}_{index}_da_seg_gt.jpg", img
                )
            if config.CLEARML_LOGGING:
                clearml_logger.current_logger().report_image(
                    "image",
                    f"{prefix}da_seg_gt{index}",
                    iteration=epoch,
                    image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                )
    return img
