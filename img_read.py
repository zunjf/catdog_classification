import tensorflow as tf
import Image
import numpy as np

filename_que = tf.train.string_input_producer(['dataset/own_dataset/cat_1.jpg'])

img_reader = tf.WholeFileReader()
key, value = img_reader.read(filename_que)

print(key)

my_img = tf.image.decode_jpeg(value)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(1):
        image = my_img.eval()

    print(image)
    Image.fromarray(np.asarray(image)).show()

    coord.request_stop()
    coord.join(threads)
