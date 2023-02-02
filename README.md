# Lidar Annotation Is All You Need
![scheme](pictures/scheme.png)
## Results

### Metrics
TODO: table from the paper

### Visualization of results
TODO: images from the paper + nice video

## Setup
### Dataset preparation
Download [waymo-open-dataset](https://github.com/waymo-research/waymo-open-dataset) (we used [1.4.0 version](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_4_0/individual_files/testing?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false)). There are training and validation folders. 2d segmentation ground truth and point cloud segmentation ground truth are made separately and not for all images. For the paper, we created a dataset from all images, for which annotations are intersected. To filter and save the dataset use this script: 
```shell
pip install -r lib/waymo_process/requirments.txt
python3 lib/waymo_process/create_2d3d_dataset.py {path_to_training_or_validation_folder_of_waymo_dataset} subset='val'
```
- --lidar_data_only=True - for saving only reprojected point cloud points for both road (gt) and other classes (loss mask)
- --masks_only=True - for saving only 2d masks.
- If no flag is chosen, you will get dataset of images where 2d segmentation ground truth and point cloud segmentation ground truth are intersected. 

You should get 1852 images in train set and 315 images in val set with both 2d masks of road and reprojected points for road and other classes. 

### Docker
Build a contatiner:
```shell
DOCKER_BUILDKIT=1 docker build --network host -t lidar_segm --target base_image --build-arg UID=1000 --build-arg GID=1000 --build-arg USERNAME={your username} .
```

Run the container:
```shell
docker run --net=host --userns=host --pid=host -itd --gpus all --name=lidar_segm --volume={path_to_lidar_data_2d_road_segmentation}:/lidar_data_2d_road_segmentation --volume={path_to_dataset}:/data/ --shm-size 15G --cpuset-cpus 0-7 lidar_segm
```

Attach to the container:
```shell
docker exec -it lidar_segm bash
```

### Conda
Alternatively you can use conda on ubuntu 20.04 with python 3.8.
```shell
conda env create -f environment.yml
```

## Training
```shell
cd /lidar_data_2d_road_segmentation
python3 scripts/train.py
```

## Testing
```shell
python scripts/test.py --weights {path to the .pth weights} --save_video
```

## Acknowledgements
* [YOLOP](https://github.com/hustvl/YOLOP)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
* [Waymo Open Dataset](https://github.com/waymo-research/waymo-open-dataset)


## Internal info (to delete later)
[Overleaf link](https://www.overleaf.com/1696216323nwdndpcgrhwx) 

[Paper concept and list of tasks with deadlines](https://evocargo.atlassian.net/wiki/spaces/PER/pages/717815826/-+Lidar+data+is+all+you+need+for+2d+road+segmentation)

### TODO:
* simplify and delete redundant:
    * config
    * loss
    * utils
* add pre-commit

### Extra:
* move to albumentations 
* move to external metrics 