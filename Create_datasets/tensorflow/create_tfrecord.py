# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import cv2
import random
import numpy as np

# 随机种子，使得每次运行结果一样
random.seed(0)


def get_files(dirpath):
    '''
    获取文件相对路径和标签（非one_hot）  返回一个元组

    args:
          dirpath：数据所在的目录 记做父目录
                  假设有10类数据，则父目录下有10个子目录，每个子目录存放着对应的图片
    '''
    # 保存读取到的的文件和标签
    image_list = []
    label_list = []

    # 遍历子目录
    classes = [x for x in os.listdir(dirpath) if os.path.isdir(dirpath)]

    # 遍历每一个子文件夹
    for index, name in enumerate(classes):
        # 子文件夹路径
        class_path = os.path.join(dirpath, name)
        # 遍历子目录下的每一个文件
        for img_name in os.listdir(class_path):
            # 每一个图片全路径
            img_path = os.path.join(class_path, img_name)
            # 追加
            image_list.append(img_path)
            label_list.append(index)

    # 保存打乱后的文件和标签
    images = []
    labels = []
    # 打乱文件顺序 连续打乱两次
    indices = list(range(len(image_list)))
    random.shuffle(indices)
    for i in indices:
        images.append(image_list[i])
        labels.append(label_list[i])
    random.shuffle([images, labels])

    print('样本长度为:', len(images))
    # print(images[0:10],labels[0:10])
    return images, labels




def WriteTFRecord(dirpath, dstpath='.', train_data=True, IMAGE_HEIGHT=227, IMAGE_WIDTH=227):
    '''
    把指定目录下的数据写入同一个TFRecord格式文件中

    args:
        dirpath：数据所在的目录 记做父目录
                 假设有10类数据，则父目录下有10个子目录，每个子目录存放着对应的图片
        dstpath:保存TFRecord文件的目录
        train_data:表示传入的是不是训练集文件所在路径
        IMAGE_HEIGHT:保存的图片数据高度
        IMAGE_WIDTH:保存的图片数据宽度
    '''
    if not os.path.isdir(dstpath):
        os.mkdir(dstpath)

    # 获取所有数据文件路径，以及对应标签
    image_list, label_list = get_files(dirpath)

    # 把海量数据写入多个TFrecord文件
    length_per_shard = 10000  # 每个记录文件的样本长度
    num_shards = int(np.ceil(len(image_list) / length_per_shard))

    print('记录文件个数：', num_shards)


    # 依次写入每一个TFRecord文件
    for index in range(num_shards):
        # 按0000n-of-0000m的后缀区分文件。n代表当前文件标号,没代表文件总数
        if train_data:
            filename = os.path.join(dstpath, 'train_data.tfrecord-%.5d-of-%.5d' % (index, num_shards))
        else:
            filename = os.path.join(dstpath, 'test_data.tfrecord-%.5d-of-%.5d' % (index, num_shards))
        print(filename)

        # 创建对象 用于向记录文件写入记录
        writer = tf.python_io.TFRecordWriter(filename)

        # 起始索引
        idx_start = index * length_per_shard
        # 结束索引
        idx_end = np.min([(index + 1) * length_per_shard - 1, len(image_list)])

        # 遍历子目录下的每一个文件
        for img_path, label in zip(image_list[idx_start:idx_end], label_list[idx_start:idx_end]):
            # 读取图像
            img = cv2.imread(img_path)

            # 缩放
            img = cv2.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))

            # 将图片转化为原生bytes
            image = img.tobytes()
            # 将数据整理成 TFRecord 需要的数据结构
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}))

            # 序列化
            serialized = example.SerializeToString()
            # 写入文件
            writer.write(serialized)
        writer.close()


def read_and_decode(filename, num_epochs=None, IMAGE_HEIGHT=227, IMAGE_WIDTH=227):
    '''
    读取TFRecord格式格式文件，返回读取到的一张图像以及对应的标签

    args:
        filename:TFRecord格式文件路径 list列表
        num_epochs:每个数据集文件迭代轮数
        IMAGE_HEIGHT:保存的图片数据高度
        IMAGE_WIDTH:保存的图片数据宽度

    '''

    filename_queue = tf.train.string_input_producer(filename, shuffle=False, num_epochs=num_epochs)
    # 创建一个文件读取器 从队列文件中读取数据
    reader = tf.TFRecordReader()

    # reader从 TFRecord 读取内容并保存到 serialized_example中
    _, serialized_example = reader.read(filename_queue)

    # 读取serialized_example的格式
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )

    # 解析从 serialized_example 读取到的内容
    img = tf.decode_raw(features['image'], tf.uint8)
    img = tf.reshape(img, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])

    '''
    在这里可以对读取到的图片数据进行预处理，比如归一化输入，PCA处理等，但是不可以增加数据    
    '''
    label = tf.cast(features['label'], tf.int32)
    return img, label


def input_data(filenames, num_epochs=None, batch_size=256, capacity=4096, min_after_dequeue=1024, num_threads=10):
    '''
    读取小批量batch_size数据

    args:
        filenames:TFRecord文件路径组成的list
        num_epochs:每个数据集文件迭代轮数
        batch_size:小批量数据大小
        capacity:内存队列元素最大个数
        min_after_dequeue：内存队列元素最小个数
        num_threads：线城数
    '''
    '''
    读取批量数据  这里设置batch_size，即一次从内存队列中随机读取batch_size张图片，这里设置内存队列最小元素个数为1024，最大元素个数为4096    
    shuffle_batch 函数会将数据顺序打乱
    bacth 函数不会将数据顺序打乱
    '''
    img, label = read_and_decode(filenames, num_epochs)
    images_batch, labels_batch = tf.train.shuffle_batch([img, label], batch_size=batch_size, capacity=capacity,
                                                        min_after_dequeue=batch_size * 5, num_threads=num_threads)
    return images_batch, labels_batch


def file_match(s, root='.'):
    '''
    寻找指定目录下（不包含子目录）中的文件名含有指定字符串的文件，并打印出其相对路径

    args:
        s：要匹配的字符串
        root : 指定要搜索的目录

    return：返回符合条件的文件列表
    '''
    # 用来保存目录
    dirs = []
    # 用来保存匹配字符串的文件
    matchs = []
    for current_name in os.listdir(root):
        add_root_name = os.path.join(root, current_name)
        if os.path.isdir(add_root_name):
            dirs.append(add_root_name)
        elif os.path.isfile(add_root_name) and s in add_root_name:
            matchs.append(add_root_name)

    '''
    #这里用来递归搜索子目录的
    for dir in dirs:
        file_match(s,dir)
    '''
    return matchs


'''
测试
'''
if __name__ == '__main__':
    # 训练集数据所在的目录
    dirpath = './data/train'

    training_step = 1

    '''    
    判断训练测试集TFRecord格式文件是否存在，不存在则生成
    如果存在，直接读取        
    '''
    # 获取当前目录下包含指定字符串的文件列表
    files = file_match('train_data.tfrecord')
    # 判断数据集是否存在
    if len(files) == 0:
        print('开始读图片文件并写入TFRecord格式文件中.........')
        # 将指定路径下所有图片存为TFRecord格式 保存到文件data.tfrecord中
        WriteTFRecord(dirpath)
        print('写入完毕!\n')
        # 正则表达式匹配
        files = tf.train.match_filenames_once('./train_data.tfrecord')

        # 读取TFRecord格式格式文件，返回读取到的batch_size图像以及对应的标签
    images_batch, labels_batch = input_data(files)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 创建一个协调器，管理线程
        coord = tf.train.Coordinator()
        # 启动QueueRunner, 此时文件名才开始进队
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print('开始训练!\n')
        for step in range(training_step):
            img, label = sess.run([images_batch, labels_batch])
            print('setp :', step)
            for i in range(256):
                cv2.imwrite('%d_%d_p.jpg' % (i, label[i]), img[i])

                # 终止线程
        coord.request_stop()
        coord.join(threads)