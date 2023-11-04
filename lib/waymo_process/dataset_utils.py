import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import tensorflow as tf

if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import camera_segmentation_utils

def get_road_seg(semantic_label, iou_thresh=0.75, min_contour_size=500):
    road = ((semantic_label == 20) | (semantic_label == 21) | (semantic_label == 22)).astype(np.uint8)
    ground = (semantic_label == 26).astype(np.uint8)
    
    cnts, hiers = cv2.findContours(road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    road_filled = np.zeros_like(road)
    road_filled = cv2.fillPoly(road_filled, cnts, color=1)
    
    cnts, hiers = cv2.findContours(ground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    for cnt in cnts:
        if cnt.shape[0] <= 4:
            continue
        contour = np.zeros_like(ground)
        contour = cv2.fillPoly(contour, [cnt], color=1)
        if (contour & road_filled).sum() == 0:
            continue
        ellipse = cv2.fitEllipse(cnt)
        ellipse_mask = np.zeros_like(ground)
        ellipse_mask = cv2.ellipse(ellipse_mask, ellipse, 1, -1)
        iou = calc_iou(contour, ellipse_mask)
        if (iou >= iou_thresh) or (contour.sum() <= min_contour_size):
            road = road | contour
    return road

def calc_iou(mask1, mask2):
    return (mask1 & mask2).sum() / (mask1 | mask2).sum()

def get_2d_road_segm_with_image(frame):
    img = frame.images[0]
    seg = img.camera_segmentation_label
    panoptic_label = camera_segmentation_utils.decode_single_panoptic_label_from_proto(seg)
    semantic_label, instance_label = camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(
        panoptic_label,
        seg.panoptic_label_divisor
    )
    segmap = get_road_seg(semantic_label)
    segmap = np.squeeze(segmap, 2) 
    return tf.image.decode_jpeg(img.image), segmap 

def save_2d_road_segm_from_frame(frame, frame_number, FILENAME, folder=Path('dataset_2d_segm'), subset='val', 
                                  visualize=False, verbose=False):

    image, segmap = get_2d_road_segm_with_image(frame)
    if visualize:
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(image)
        axarr[1].imshow(segmap)

    folder_to_save_images = folder / 'images' / subset
    folder_to_save_images.mkdir(parents=True, exist_ok=True)
    cv2.imwrite((folder_to_save_images / f'{FILENAME.name}_{"0" * (5 - len(str(frame_number)))}{frame_number}.jpg').as_posix(), 
                cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

    folder_to_save_seg_mask = folder / 'seg_masks' / subset
    folder_to_save_seg_mask.mkdir(parents=True, exist_ok=True)
    cv2.imwrite((folder_to_save_seg_mask / f'{FILENAME.name}_{"0" * (5 - len(str(frame_number)))}{frame_number}.png').as_posix(), segmap * 255)
    if verbose:
        print('2d seg mask is saved')

def convert_range_image_to_point_cloud_labels(frame,
                                              range_images,
                                              segmentation_labels,
                                              ri_index=0):
    """Convert segmentation labels from range images to point clouds.

    Args:
    frame: open dataset frame
    range_images: A dict of {laser_name, [range_image_first_return,
       range_image_second_return]}.
    segmentation_labels: A dict of {laser_name, [range_image_first_return,
       range_image_second_return]}.
    ri_index: 0 for the first return, 1 for the second return.

    Returns:
    point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
      points that are not labeled.
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    point_labels = []
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0

        if c.name in segmentation_labels:
            sl = segmentation_labels[c.name][ri_index]
            sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
            sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
        else:
            num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
            sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)

        point_labels.append(sl_points_tensor.numpy())
    return point_labels

def rgba(r):
    """Generates a color based on range.

    Args:
    r: the range value of a given point.
    Returns:
    The color for a given range
    """
    c = plt.get_cmap('jet')((r % 20.0) / 20.0)
    c = list(c)
    c[-1] = 0.5  # alpha
    return c

def plot_image(camera_image):
    """Plot a cmaera image."""
    plt.figure(figsize=(20, 12))
    plt.imshow(tf.image.decode_jpeg(camera_image.image))

def plot_points_on_image_upd(projected_points, camera_image, rgba_func,
                         point_size=5.0, visualize=True):
    """Plots points on a camera image.

    Args:
    projected_points: [N, 3] numpy array. The inner dims are
      [camera_x, camera_y, range].
    camera_image: jpeg encoded camera image.
    rgba_func: a function that generates a color from a range value.
    point_size: the point size.

    """
    if visualize:
        plot_image(camera_image)

        xs = []
        ys = []
        colors = []

        for point in projected_points:
            xs.append(point[0])  # width, col
            ys.append(point[1])  # height, row
            colors.append(rgba_func(point[2]))

        plt.scatter(xs, ys, c=colors, s=point_size, edgecolors="none")

    return tf.image.decode_jpeg(camera_image.image), projected_points

def visualize_segm_pointcloud_on_image(frame, cp_points_all, points_all, img_id=0, visualize=True):
    images = sorted(frame.images, key=lambda i:i.name)
    cp_points_all_concat = np.concatenate([cp_points_all, points_all], axis=-1)
    cp_points_all_concat_tensor = tf.constant(cp_points_all_concat)

    # The distance between lidar points and vehicle frame origin.
    points_all_tensor = tf.norm(points_all, axis=-1, keepdims=True)
    cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

    mask = tf.equal(cp_points_all_tensor[..., 0], images[img_id].name)

    cp_points_all_tensor = tf.cast(tf.gather_nd(
        cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
    points_all_tensor = tf.gather_nd(points_all_tensor, tf.where(mask)) # for distance

    projected_points_all_from_raw_data = tf.concat(
        [cp_points_all_tensor[..., 1:3], points_all_tensor], axis=-1).numpy()
    
    image, points = plot_points_on_image_upd(projected_points_all_from_raw_data,
                     images[img_id], rgba, point_size=5.0, visualize=visualize)

    return image, points

def filter_and_save_data(frame, point_labels_all, cp_points_all, points_all, 
                         frame_number, FILENAME, folder=Path('dataset_lidar_segm'), 
                         visualize=False, subset='val', save_images=True):
    # only road
    cp_points_all_filtered_road = cp_points_all[(point_labels_all[:, 1] == 18) | 
                                           (point_labels_all[:, 1] == 19) |
                                            (point_labels_all[:, 1] == 20)]
    points_all_input_filtered_road = points_all[(point_labels_all[:, 1] == 18) | 
                                           (point_labels_all[:, 1] == 19) |
                                           (point_labels_all[:, 1] == 20)]

    ind = 0 # TYPE_UNDEFINED
    cp_points_all_filtered = cp_points_all[(point_labels_all[:, 1] != ind)]
    points_all_input_filtered = points_all[(point_labels_all[:, 1] != ind)]

    image, points_road = visualize_segm_pointcloud_on_image(frame, cp_points_all_filtered_road, 
                                   points_all_input_filtered_road, 0, visualize=visualize)

    _, points_all = visualize_segm_pointcloud_on_image(frame, cp_points_all_filtered, 
                                   points_all_input_filtered, 0, visualize=visualize)
    
    if save_images:
        folder_to_save_images = folder / 'images' / subset
        folder_to_save_images.mkdir(parents=True, exist_ok=True)
        cv2.imwrite((folder_to_save_images / f'{FILENAME.name}_{"0" * (5 - len(str(frame_number)))}{frame_number}.jpg').as_posix(), cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

    points_xy_road = points_road[:, :2]
    points_int_road = (points_xy_road[:, [1, 0]]).astype(int)
    folder_to_save_points_road = folder / 'seg_points' / subset
    folder_to_save_points_road.mkdir(parents=True, exist_ok=True)
    np.save((folder_to_save_points_road / f'{FILENAME.name}_{"0" * (5 - len(str(frame_number)))}{frame_number}.npy').as_posix(), points_int_road)

    points_xy = points_all[:, :2]
    points_int = (points_xy[:, [1, 0]]).astype(int)
    folder_to_save_points_all = folder / 'seg_points_total' / subset
    folder_to_save_points_all.mkdir(parents=True, exist_ok=True)
    np.save((folder_to_save_points_all / f'{FILENAME.name}_{"0" * (5 - len(str(frame_number)))}{frame_number}.npy').as_posix(), points_int)

def get_3d_data_from_frame(frame, frame_number, FILENAME, folder=Path('dataset_lidar_segm'), 
                           visualize=False, subset='val', save_images=True, verbose=False):
    (range_images, camera_projections, segmentation_labels,
     range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
        frame)

    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose)
    points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose, ri_index=1)

    point_labels = convert_range_image_to_point_cloud_labels(
        frame, range_images, segmentation_labels)
    point_labels_ri2 = convert_range_image_to_point_cloud_labels(
        frame, range_images, segmentation_labels, ri_index=1)

    # 3d points in vehicle frame
    points_all = np.concatenate(points, axis=0)
    _points_all_ri2 = np.concatenate(points_ri2, axis=0)
    # point labels
    point_labels_all = np.concatenate(point_labels, axis=0)
    _point_labels_all_ri2 = np.concatenate(point_labels_ri2, axis=0)
    # camera projection corresponding to each point
    cp_points_all = np.concatenate(cp_points, axis=0)
    _cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)

    filter_and_save_data(frame, point_labels_all, cp_points_all, points_all, frame_number, 
                         FILENAME, folder, visualize, subset, save_images)
    if verbose:
        print('3d reprojected points is saved')