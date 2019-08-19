#coding=utf-8

import tensorflow as tf 
import numpy as np 
import pdb
import cv2
import os
import glob
import slim.nets.resnet_v1 as resnet_v1

import tensorflow.contrib.slim as slim

import create_csv_files

os.environ["CUDA_VISIBLE_DEVICES"]='1'

def read_image(filename, resize_height, resize_width,normalization=False):
    '''
    读取图片数据,默认返回的是uint8,[0,255]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param normalization:是否归一化到[0.,1.0]
    :return: 返回的图片数据
    '''

    bgr_image = cv2.imread(filename)
    if len(bgr_image.shape)==2:#若是灰度图则转为三通道
        print("Warning:gray image",filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)#将BGR转为RGB
    # show_image(filename,rgb_image)
    # rgb_image=Image.open(filename)
    if resize_height>0 and resize_width>0:
        rgb_image=cv2.resize(rgb_image,(resize_width,resize_height))
    rgb_image=np.asanyarray(rgb_image)
    if normalization:
        # 不能写成:rgb_image=rgb_image/255
        rgb_image=rgb_image/255.0
    # show_image("src resize image",image)
    return rgb_image


def  predict(models_path,image_dir,labels_filename,labels_nums, data_format):
    [batch_size, resize_height, resize_width, depths] = data_format

    #labels = np.loadtxt(labels_filename, str, delimiter='\t')
    input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')

    #其他模型预测请修改这里
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        out, end_points = resnet_v1.resnet_v1_101(inputs=input_images, num_classes=labels_nums,is_training=False)

    # 将输出结果进行softmax分布,再求最大概率所属类别
    score = tf.nn.softmax(out,name='pre')
    class_id = tf.argmax(score, 1)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, models_path)
    images_list=glob.glob(os.path.join(image_dir,'*.jpg'))
    for image_path in images_list:
        im=read_image(image_path,resize_height,resize_width,normalization=True)
        im=im[np.newaxis,:]
        #pred = sess.run(f_cls, feed_dict={x:im, keep_prob:1.0})
        pre_score,pre_label = sess.run([score,class_id], feed_dict={input_images:im})
        max_score=pre_score[0,pre_label]
        print("{} is: pre labels:{},name:{} score: {}".format(image_path,pre_label,list(labels_filename.keys())[list(labels_filename.values()).index(pre_label)], max_score))
    sess.close()


if __name__ == '__main__':
    dataset_path = './dataset/'
    # num of class
    _, dirnames, class_nums = create_csv_files.get_number_of_classification(dataset_path + 'train/')
    labels_filename = create_csv_files.get_classification_label(dirnames, class_nums)

    image_dir = './test_image'
    # labels_filename='dataset/label.txt'
    models_path = 'models_resnet/model.ckpt-10000'

    batch_size = 1  #
    resize_height = 299  # 指定存储图片高度
    resize_width = 299  # 指定存储图片宽度
    depths = 3
    data_format = [batch_size, resize_height, resize_width, depths]
    predict(models_path, image_dir, labels_filename, class_nums, data_format)
