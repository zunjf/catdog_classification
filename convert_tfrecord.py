import tensorflow as tf
import os

path_data = 'dataset/train'

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(path_folder, output_file):
    output_file = os.path.join('dataset/'+output_file+'.tfrecords')

    print('Writing '+output_file)
    writer = tf.python_io.TFRecordWriter(output_file)

    ls_files = os.listdir(path_folder)

    # rows = images.shape[1]
    # cols = images.shape[2]
    # depth = images.shapes[3]

    for fl in ls_files:
        # img_raw = read file
        example = tf.train.Example(features=tf.train.Features(feature={
            'height' : _int64_feature(rows)
        }))