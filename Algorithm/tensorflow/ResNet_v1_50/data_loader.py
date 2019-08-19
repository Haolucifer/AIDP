import tensorflow as tf
import create_csv_files

IMG_CHANNELS = 3

IMG_HEIGHT = 299
IMG_WIDTH = 299

batch = 16


#提取图像及标签信息
def _load_samples(csv_name, image_type):
    filename_queue = tf.train.string_input_producer(
        [csv_name])

    reader = tf.TextLineReader()
    _, csv_filename = reader.read(filename_queue)

    record_defaults = [tf.constant([], dtype=tf.string),
                       tf.constant([], dtype=tf.int32)]

    filename_i, label = tf.decode_csv(
        csv_filename, record_defaults=record_defaults)

    file_contents = tf.read_file(filename_i)
    if image_type == '.jpg':
        image_decoded = tf.image.decode_jpeg(
            file_contents, channels=IMG_CHANNELS)
    elif image_type == '.png':
        image_decoded = tf.image.decode_png(
            file_contents, channels=IMG_CHANNELS, dtype=tf.uint8)

    return image_decoded, label


def load_data(dataset_csv_path, image_type, image_size_before_crop, labels_nums,
              do_shuffle=True, do_flipping=False, Normalization=True, one_hot=True):
    """
    :param dataset_csv_path: The path of the csv_dataset.
    :param image_type:the type of image, default='.jpg'
    :param image_size_before_crop: Resize to this size before random cropping.
    :param do_shuffle: Shuffle switch.
    :param do_flipping: Flip switch.
    :param Normalization:default=True
    :param one_hot:default=True
    :return:
    """
    image, label = _load_samples(dataset_csv_path,image_type)
    inputs = {
        'image': image,
        'label': label
    }

    # Preprocessing:
    inputs['image'] = tf.image.resize_images(
        inputs['image'], [image_size_before_crop, image_size_before_crop])
    if do_flipping is True:
        inputs['image'] = tf.image.random_flip_left_right(inputs['image'])
    inputs['image'] = tf.random_crop(
        inputs['image'], [IMG_HEIGHT, IMG_WIDTH, 3])
    #Normalization
    if Normalization:
        inputs['image'] = tf.cast(inputs['image'], tf.float32) * (1. / 255.0)
    else:
        inputs['image'] = tf.cast(inputs['image'], tf.float32)
        
    # Batch
    if do_shuffle is True:
        inputs['image'], inputs['label'] = tf.train.shuffle_batch(
            [inputs['image'], inputs['label']], batch, 5000, 100)
    else:
        inputs['image'], inputs['label'] = tf.train.batch(
            [inputs['image'], inputs['label']], batch)
        
    if one_hot:
        inputs['label'] = tf.one_hot(inputs['label'],labels_nums,1,0)

    return inputs

if __name__ == '__main__':
    data_csv_path_train = './dataset/train/train.csv'
    image_type = '.jpg'
    resize_height = 224
    labels_nums = 5
    train_batch = load_data(data_csv_path_train,image_type=image_type,
                                                                   image_size_before_crop=resize_height, labels_nums=labels_nums)
    train_images_batch = train_batch['image']
    train_labels_batch = train_batch['label']
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        for i in range(10):
            batch_input_images, batch_input_labels = sess.run([train_images_batch, train_labels_batch])
            print(batch_input_images,batch_input_labels)
        
        coord.request_stop()
        coord.join(threads)