import tensorflow as tf
from matplotlib import pyplot as plt

filename_que = tf.train.string_input_producer(['dataset/own_dataset/cat_1.jpg'])

img_reader = tf.WholeFileReader()
key, value = img_reader.read(filename_que)

print(key)

my_img = tf.image.decode_jpeg(value)
my_img = tf.image.resize_images(my_img, [100, 100])

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(1):
        image = my_img.eval()
        plt.imshow(image, interpolation='nearest')
        plt.show()
        plt.close()

    coord.request_stop()
    coord.join(threads)
