# -*-coding:utf-8-*-
import os
import os.path

import csv

import random


#解析文件夹下子文件夹数目
def get_number_of_classification(filepath):
    for path, dirnames, _ in os.walk(filepath):
        break
    for dirname in dirnames:
        if '.ipynb' in dirname:
            dirnames.remove(dirname)
    dirnames.sort()
#     print('nihao')
#     print(dirnames, len(dirnames))
    return path, dirnames, len(dirnames)

#给每个类别定义标签
def get_classification_label(dirnames, num_of_labels):
    classification_label = {}
    num = 0
    for dirname in dirnames:
        classification_label[dirname]=num
        num = num+1
    if num > num_of_labels:
        raise ValueError('the number of labels is false, check it!')
    return classification_label

#写入csv文件
def write_to_csv(source_filepath,target_filepath):
    root, dirnames, num_of_labels = get_number_of_classification(source_filepath)
    classification_label = get_classification_label(dirnames, num_of_labels)
    print(classification_label)
    file_list = []
    for dirname in dirnames:
        labels = classification_label[dirname]
        file_names = os.listdir(os.path.join(root,dirname))
        for file_name in file_names:
            file_list.append((os.path.join(root,dirname,file_name), labels))
    #shuffle the file_list
    random.shuffle(file_list)
    with open(target_filepath,'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        for file in file_list:
            csv_writer.writerow(list(file))
    _, name = os.path.split(target_filepath)
    print('write {} successifully'.format(name))


def main():
    write_to_csv('./dataset/train/','./dataset/train/train.csv')
    write_to_csv('./dataset/val', './dataset/val/val.csv')

if __name__ == '__main__':
    main()





