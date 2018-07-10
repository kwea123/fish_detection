# fish_detection

the Open Images Dataset is downloadable from [cvdfoundation](https://github.com/cvdfoundation/open-images-dataset#download-images-with-bounding-boxes-annotations), of size 513GB.

Among all images, there are `24403` individual fish bounding boxes according to [read_fish.ipynb](read_fish.ipynb).

## Tensorflow Object Detection API Setup

Please execute the following commands from `models/research/` :
```
protoc object_detection/protos/*.proto --python_out=. # protoc needs to be version 3
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
These are the reasons if you get `ImportError: No module named deployment` or `ImportError: No module named object_detection` when training.
