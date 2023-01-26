# Lidar Annotation Is All You Need

[Overleaf link](https://www.overleaf.com/1696216323nwdndpcgrhwx) 

[Paper concept and list of tasks with deadlines](https://evocargo.atlassian.net/wiki/spaces/PER/pages/717815826/-+Lidar+data+is+all+you+need+for+2d+road+segmentation)

### TODO:
* run experiments for all cases (same as for YOLOP experiments)
* add requirements.txt or Dockerfile
* add script for waymo dataset processing
* simplify and delete redundant:
    * config
    * loss
    * utils
* add pre-commit

### Extra:
* move to albumentations 
* move to external metrics 

### Acknowledgements
* [YOLOP](https://github.com/hustvl/YOLOP)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
* [Waymo Open Dataset](https://github.com/waymo-research/waymo-open-dataset)