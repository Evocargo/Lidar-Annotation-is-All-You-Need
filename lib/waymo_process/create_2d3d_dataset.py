from os import listdir
from os.path import isfile, join
from pathlib import Path

import fire
import tensorflow as tf
from dataset_utils import get_3d_data_from_frame, save_2d_road_segm_from_frame
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2 as open_dataset


if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()


def save_dataset_from_records(
    records_path,
    lidar_data_only=False,
    masks_only=False,
    subset="val",
    save_folder="waymo_2d_3d_segm",
    verbose=False,
):
    print(records_path)
    records_list = [
        Path(records_path) / f
        for f in listdir(records_path)
        if isfile(join(records_path, f))
    ]

    total_frames = 0
    corrupted_frames = 0
    for FILENAME in tqdm(records_list, total=len(records_list)):
        print(f"record - {FILENAME}")
        dataset = tf.data.TFRecordDataset(FILENAME, compression_type="")
        frames_saved = 0
        for frame_number, data in enumerate(dataset):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if lidar_data_only:
                if frame.lasers[0].ri_return1.segmentation_label_compressed:
                    frames_saved += 1
                    get_3d_data_from_frame(
                        frame,
                        frame_number,
                        FILENAME,
                        folder=Path(save_folder),
                        subset=subset,
                        save_images=True,
                        verbose=verbose,
                    )
            elif masks_only:
                if frame.images[0].camera_segmentation_label.ByteSize() != 0:
                    frames_saved += 1
                    try:
                        save_2d_road_segm_from_frame(
                            frame,
                            frame_number,
                            FILENAME,
                            folder=Path(save_folder),
                            subset=subset,
                            verbose=verbose,
                        )
                    except Exception as e:
                        print(f"Frame is corrupted with exception: {e}")
                        corrupted_frames += 1
                        continue

            else:
                if (
                    frame.images[0].camera_segmentation_label.ByteSize() != 0
                    and frame.lasers[0].ri_return1.segmentation_label_compressed
                ):
                    frames_saved += 1
                    save_2d_road_segm_from_frame(
                        frame,
                        frame_number,
                        FILENAME,
                        folder=Path(save_folder),
                        subset=subset,
                        verbose=verbose,
                    )
                    get_3d_data_from_frame(
                        frame,
                        frame_number,
                        FILENAME,
                        folder=Path(save_folder),
                        subset=subset,
                        save_images=False,
                        verbose=verbose,
                    )
            if verbose:
                print(f"saved frame #{frames_saved }")
        print(f"Saved {frames_saved} frames from this record")
        total_frames += frames_saved
    print(f"Saved {total_frames} frames in total")
    print(f"Corrupted frames: {corrupted_frames}")


if __name__ == "__main__":
    fire.Fire(save_dataset_from_records)
