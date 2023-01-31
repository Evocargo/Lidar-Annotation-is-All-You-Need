# Lidar Annotation Is All You Need

[Overleaf link](https://www.overleaf.com/1696216323nwdndpcgrhwx) 

[Paper concept and list of tasks with deadlines](https://evocargo.atlassian.net/wiki/spaces/PER/pages/717815826/-+Lidar+data+is+all+you+need+for+2d+road+segmentation)

### Setup

#### Docker
TODO

#### Conda
```shell
conda env create -f environment.yml
```

### Dataset preparation
TODO

### Train
```shell
python3 scripts/train.py
```

### Test
```shell
python scripts/test.py --weights {path to the .pth weights} --save_video
```

### Acknowledgements
* [YOLOP](https://github.com/hustvl/YOLOP)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
* [Waymo Open Dataset](https://github.com/waymo-research/waymo-open-dataset)

### TODO:
* add script for waymo dataset processing
* simplify and delete redundant:
    * config
    * loss
    * utils
* add pre-commit

### Extra:
* move to albumentations 
* move to external metrics 