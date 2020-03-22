"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('images_path', '', 'Path to images files')
FLAGS = flags.FLAGS


def class_text_to_int(row_label):
    if row_label=="Damage":
        return 1
    else:
        return None
def getDicts(row_iterator):
    _dict=None
    for index, row in row_iterator.iterrows():
        if _dict is None or _dict["filename"]!=row["filename"]:
            if _dict is not None:
                yield _dict
            _dict=dict(row)
            for key in ["xmin","xmax","ymin","ymax","class"]:
                _dict[key]=[row[key]]
        else:
            for key in ["xmin","xmax","ymin","ymax","class"]:
                _dict[key].append(row[key])
    if _dict is not None:
        yield _dict
def create_tf_example(_dict):
    # print(_dict)
    row=_dict #to avoid renaming elsewhere in code
    img_path=FLAGS.images_path
    full_path = os.path.join(os.getcwd(), img_path, '{}'.format(row['filename']))
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = row['filename'].encode('utf8')
    image_format = b'jpg'
    xmins = [r / width for r in row['xmin']]
    xmaxs = [r / width for r in row['xmax']]
    ymins = [r / height for r in row['ymin']]
    ymaxs = [r / height for r in row['ymax']]
    classes_text = [r.encode('utf8') for r in row["class"]]
    classes = [class_text_to_int(r) for r in row["class"]]
    # print(xmins)
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

import logging
def main(_):

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    examples = pd.read_csv(FLAGS.csv_input)
    
    for _dict in getDicts(examples):
        try:
            tf_example = create_tf_example(_dict)
            writer.write(tf_example.SerializeToString())
            logging.warning("Successfull wrote tf record for {}".format(_dict))
        except:
            logging.exception("Failed to write tf record for {}".format(_dict))
    
    writer.close()


if __name__ == '__main__':
    tf.app.run()
