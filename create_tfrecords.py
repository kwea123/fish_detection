# Borrowed from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
# Run 'python create_tfrecords.py --output_path data/fish_train.record --csv_input data/fish_train.csv --image_path=root/data/images/train/'

import tensorflow as tf
from utils import dataset_util
import pandas as pd
from PIL import Image
import os
import numpy as np

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to csv file')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_path', '', 'Path to the images')
FLAGS = flags.FLAGS

def create_tf_example(image_path, example, image_id):
    """
    Create a tf_example using @example.
    @example is of form : ["ImageID", "XMin", "XMax", "YMin", "YMax"] which are the columns of "fish.csv".
    @example contains all bounding boxes for the image with @image_id
    """
    
    filename = image_id+'.jpg'
    
    image_path = os.path.join(image_path, filename)
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_image_data = fid.read()
        
    filename = filename.encode()
    image = Image.open(image_path)
    width, height = image.size
    del image
    
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    
    for bbox in np.array(example[['XMin', 'XMax', 'YMin', 'YMax']]):
        xmins += [bbox[0]]
        xmaxs += [bbox[1]]
        ymins += [bbox[2]]
        ymaxs += [bbox[3]]
    
    classes_text = [b'fish']*len(example)
    classes = [1]*len(example)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    
    df = pd.read_csv(FLAGS.csv_input)
    image_id_list = list(set(df['ImageID']))
    n_examples = len(image_id_list)

    for i, image_id in enumerate(image_id_list):
        tf_example = create_tf_example(FLAGS.image_path, df[df['ImageID']==image_id], image_id)
        writer.write(tf_example.SerializeToString())
        print('\rprocessing %d of all %d images'%(i+1, n_examples), end="")
    print('\nDone!')
    writer.close()

if __name__ == '__main__':
    tf.app.run()
