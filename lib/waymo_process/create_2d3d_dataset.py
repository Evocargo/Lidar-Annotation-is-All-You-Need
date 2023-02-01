from pathlib import Path
from os import listdir
from os.path import isfile, join
import fire
from dataset_utils import save_2d_road_segm_from_frame, get_3d_data_from_frame
from waymo_open_dataset import dataset_pb2 as open_dataset

import tensorflow as tf
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

def save_dataset_from_records(records_path, lidar_data_only=False, masks_only=False, 
                              subset='val', save_folder='test_dataset', verbose=False):
    print(records_path)
    records_list = [Path(records_path) / f for f in listdir(records_path) if isfile(join(records_path, f))]
    
    total_frames = 0
    for FILENAME in records_list:
        print(f'record - {FILENAME}')
        dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
        frame_number = -1
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if lidar_data_only:
                if frame.lasers[0].ri_return1.segmentation_label_compressed:
                    frame_number += 1
                    get_3d_data_from_frame(frame, frame_number, FILENAME, folder=Path(save_folder), subset=subset, 
                                                save_images=True, verbose=verbose)
            elif masks_only:
                if frame.images[0].camera_segmentation_label.ByteSize() != 0:
                    frame_number += 1
                    save_2d_road_segm_from_frame(frame, frame_number, FILENAME, folder=Path(save_folder), 
                                                 subset=subset, verbose=verbose)
            else:
                if frame.images[0].camera_segmentation_label.ByteSize() != 0 and frame.lasers[0].ri_return1.segmentation_label_compressed:
                    frame_number += 1
                    save_2d_road_segm_from_frame(frame, frame_number, FILENAME, folder=Path(save_folder), 
                                                 subset=subset, verbose=verbose)
                    get_3d_data_from_frame(frame, frame_number, FILENAME, folder=Path(save_folder), subset=subset, 
                                                save_images=False, verbose=verbose)
            if verbose:
                print(f'saved frame #{frame_number + 1}')
        print(f'Saved {frame_number + 1} frames from this record')
        total_frames += frame_number + 1
    print(f'Saved {total_frames} frames in total')

if __name__ == '__main__':
    fire.Fire(save_dataset_from_records)