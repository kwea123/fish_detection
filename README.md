# fish_detection

the Open Images Dataset is downloadable from [cvdfoundation](https://github.com/cvdfoundation/open-images-dataset#download-images-with-bounding-boxes-annotations), of size 513GB.

Among all images, there are `24403` individual fish bounding boxes according to [read_fish.ipynb](read_fish.ipynb).

## Tensorflow Object Detection API Setup

Clone Tensorflow Object Detection github :
```
git clone https://github.com/tensorflow/models/tree/master/research/object_detection
```

Execute the following commands from `models/research/` :
```
protoc object_detection/protos/*.proto --python_out=. # protoc needs to be version 3
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim # needs to be executed each time in a new shell
```
Normally, `protoc 3` should be installed when you build the docker.
If not, refer to [this page](https://gist.github.com/sofyanhadia/37787e5ed098c97919b8c593f0ec44d8) and install it (`protoc 2` **doesn't work**).

If you forget to run them, you could get `ImportError: No module named deployment` or `ImportError: No module named object_detection` when training.

## Start training
From `models/research/`, run
```
python object_detection/train.py --logtostderr --train_dir=${YOUR MODEL'S OUTPUT DIR} --pipeline_config_path=${YOUR CONFIG's PATH} 
```
The paths should be **absolute**!

You will get the following training info :
```
INFO:tensorflow:global step 1758: loss = 6.4747 (0.338 sec/step)
INFO:tensorflow:global step 1758: loss = 6.4747 (0.338 sec/step)
2018-07-10 14:57:50.831791: W tensorflow/core/framework/allocator.cc:101] Allocation of 226492416 exceeds 10% of system memory.
INFO:tensorflow:global step 1759: loss = 5.3687 (0.348 sec/step)
INFO:tensorflow:global step 1759: loss = 5.3687 (0.348 sec/step)
2018-07-10 14:57:51.179552: W tensorflow/core/framework/allocator.cc:101] Allocation of 301989888 exceeds 10% of system memory.
INFO:tensorflow:global step 1760: loss = 5.8365 (0.305 sec/step)
INFO:tensorflow:global step 1760: loss = 5.8365 (0.305 sec/step)
2018-07-10 14:57:51.488248: W tensorflow/core/framework/allocator.cc:101] Allocation of 301989888 exceeds 10% of system memory.
```

Your model is on the way!
