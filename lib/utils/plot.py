import cv2
import numpy as np
import random

def show_seg_result(
    img, 
    result, 
    index, 
    epoch, 
    save_dir=None, 
    is_ll=False, 
    palette=None, 
    is_demo=False, 
    is_gt=False, 
    config=None, 
    clearml_logger=None):

    if palette is None:
        palette = np.random.randint(
                0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)
    assert palette.shape[0] == 3 # len(classes)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    
    if not is_demo:
        color_seg = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[result == label, :] = color
    else:
        color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)
        
        # for label, color in enumerate(palette):
        #     color_area[result[0] == label, :] = color

        color_area[result[0] == 1] = [0, 255, 0]
        color_area[result[1] ==1] = [255, 0, 0]
        color_seg = color_area

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    color_mask = np.mean(color_seg, 2)

    # color_mask = cv2.resize(color_mask, (img.shape[1], img.shape[0]))
    # color_seg = cv2.resize(color_seg, (img.shape[1], img.shape[0]))
    # img = cv2.resize(img, (color_mask.shape[1], color_mask.shape[0]))

    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    # img = img * 0.5 + color_seg * 0.5
    #img = img.astype(np.uint8)
    # FIX idk why it was here, commented resize
    # img = cv2.resize(img, (1280,720), interpolation=cv2.INTER_LINEAR)

    if not is_demo:
        if not is_gt:
            if not is_ll:
                if config.TRAIN.SAVE_LOCALLY:
                    cv2.imwrite(save_dir+"/batch_{}_{}_da_segresult.jpg".format(epoch, index), img)
                if config.TRAIN.CLEARML_LOGGING:
                    clearml_logger.current_logger().report_image(
                        "image", f"da_segresult{index}", iteration=epoch, image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                if config.TRAIN.SAVE_LOCALLY:
                    cv2.imwrite(save_dir+"/batch_{}_{}_ll_segresult.jpg".format(epoch, index), img)
                if config.TRAIN.CLEARML_LOGGING:
                    clearml_logger.current_logger().report_image(
                        "image", f"ll_segresult{index}", iteration=epoch, image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            if not is_ll:
                if config.TRAIN.SAVE_LOCALLY:
                    cv2.imwrite(save_dir+"/batch_{}_{}_da_seg_gt.jpg".format(epoch, index), img)
                if config.TRAIN.CLEARML_LOGGING:
                    clearml_logger.current_logger().report_image(
                        "image", f"da_seg_gt{index}", iteration=epoch, image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                if config.TRAIN.SAVE_LOCALLY:
                    cv2.imwrite(save_dir+"/batch_{}_{}_ll_seg_gt.jpg".format(epoch, index), img)  
                if config.TRAIN.CLEARML_LOGGING:
                    clearml_logger.current_logger().report_image(
                        "image", f"ll_seg_gt{index}", iteration=epoch, image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.0001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 5, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 5, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


if __name__ == "__main__":
    pass