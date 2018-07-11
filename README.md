# fish_detection

# Data preparation

the Open Images Dataset is downloadable from [cvdfoundation](https://github.com/cvdfoundation/open-images-dataset#download-images-with-bounding-boxes-annotations), of size 513GB.

Download all the images along with the annotations.

Clone this repository with `git clone https://github.com/kwea123/fish_detection.git`.

# Setup the environment

A `Dockerfile` with all dependencies is provided. You can build it with
```
nvidia-docker build $CONTAINER:$TAG ./
```

Otherwise, an environment with all dependencies also works.

# Examine the data

## Check and extract data

To check how the data format looks like, see [read_test.ipynb](read_test.ipynb).

Since we only want to train on fish (and related species), use [read_fish.ipynb](read_fish.ipynb) to see how much data we actually have.

Among all images, there are `24403` individual fish bounding boxes training data.

Also, we save the minimum required data (ImageId and bounding box coordinates) into 'fish_train.csv' and 'fish_val.csv' (you can use the test set too).

# Choose an object detection model

I choose [Tensorflow Object Detection](https://github.com/tensorflow/models/tree/master/research/object_detection) to be my detection model.

## 1. Setup Tensorflow Object Detection API

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

## 2. Prepare TFRecord

In order to train a tensorflow model, we need to prepare the data in its acceptable form, which are `tfrecord`s.

Following the official [tutorial](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md), I created [create_tfrecords.py](create_tfrecords.py) which converts the `.csv` files created in [read_fish.ipynb](read_fish.ipynb) into '.record' files.

Run
```
python create_tfrecords.py --output_path data/fish_train.record --csv_input data/fish_train.csv --image_path=/root/data/images/train/
```
with the paths set correctly to your paths.

## 3. Create the label map

From `/models/research/object_detection/data`, you can see sample label maps.

Create your own according to your classes. E.g. mine is [fish_label_map.pbtxt](data/fish_label_map.pbtxt)

**Note** : The class id must start from **1**.

Now the data prepartion is completed. We move on to prepare the model.

## 4. Download an existing model

Download a model from [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), and extract the `.tar` file.

I use [faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz) since it's fast and enough accurate.

**Note** : I had difficulty obtaining good results using ssd nets.
See [Issue](https://github.com/tensorflow/models/issues/3196) for more information.

## 5. Pick the corresponding config

Pick the model's config from `/models/research/object_detection/samples/configs`. Duplicate it somewhere.

**Note** : you should pick the config with the **same** name as your model.

## 6. Modify the config

Open the duplicated config, (change its name if you wish) and modify the following according to your model:

* `num_classes`, which should be at the beginning of the file;
* `fine_tune_checkpoint` and
* 
```
train_input_reader: {
  tf_record_input_reader {
    input_path: "PATH_TO_BE_CONFIGURED/mscoco_train.record"
  }
  label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"
}

eval_config: {
  num_examples: 8000
  # Note: The below line limits the evaluation process to 10 evaluations.
  # Remove the below line to evaluate indefinitely.
  max_evals: 10
}

eval_input_reader: {
  tf_record_input_reader {
    input_path: "PATH_TO_BE_CONFIGURED/mscoco_val.record"
  }
  label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"
  shuffle: false
  num_readers: 1
}
```
which are towards the end of the file.

**Note** : The paths should be **absolute**!

You can refer to [my configuration](faster_rcnn_inception_v2_coco.config).

## 6. Start training

You can start training now!

From `models/research/`, run
```
python object_detection/train.py --logtostderr --train_dir=${YOUR MODEL'S OUTPUT DIR} --pipeline_config_path=${YOUR CONFIG's PATH} 
```
**Note** : The paths should be **absolute**!

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
