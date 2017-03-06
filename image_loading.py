import tensorflow as tf

filename_que = tf.train.string_input_producer(
    tf.train.match_filenames_once("dataset/*.jpg"))

image_reader = tf.WholeFileReader()

_, image_file = image_reader.read(filename_que)

image = tf.image.decode_jpeg(image_file)
image = tf.image.resize_images(image, [224, 224])

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    image_tensor = sess.run([image])
    print (image_tensor)

    coord.request_stop()
    coord.join(threads)
