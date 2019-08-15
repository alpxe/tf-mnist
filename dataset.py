"""
将mnist gz文件构建成 TFRecord
"""
import gzip
import os
import struct
import numpy as np
import tensorflow as tf
import sys

TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

MNIST_DATA = "MNIST_DATA"
TFrecords = "tfrecords"


def create(IMAGES, LABELS, NAME):
    iamge_path = os.path.join(MNIST_DATA, IMAGES)
    if os.path.exists(iamge_path):
        with open(iamge_path, 'rb') as f:  # 打开文件
            with gzip.GzipFile(fileobj=f)as bytestream:  # 解压缩
                img_buf = bytestream.read()

    label_path = os.path.join(MNIST_DATA, LABELS)
    if os.path.exists(label_path):
        with open(label_path, 'rb') as f:
            with gzip.GzipFile(fileobj=f)as bytestream:
                label_buf = bytestream.read()

    # 获取字节流中，前4个字符。<>,<样本数量>,<样本纬度>,<样本纬度>
    magic, items, row, col = struct.unpack_from(">IIII", img_buf, 0)

    img_hd = struct.calcsize(">IIII")
    label_hd = struct.calcsize(">II")

    if not os.path.exists(TFrecords):
        os.makedirs(TFrecords)  # 生成文件夹

    # 生成 .tfrecords
    writer = tf.python_io.TFRecordWriter(os.path.join(TFrecords, NAME))

    for i in range(items):  # 60000个样本
        label, = struct.unpack_from(">B", label_buf, label_hd + i * 1)
        img = struct.unpack_from(">{0}B".format(row * col), img_buf, img_hd + i * row * col)

        fs = tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            "image": tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(img)))
        })
        example = tf.train.Example(features=fs)
        writer.write(example.SerializeToString())  # SerializeToString 写入

        sys.stdout.write("\rWrite progress： {0:.2f}%".format((i+1)/items*100))
        sys.stdout.flush()

    writer.close()



create(TRAIN_IMAGES, TRAIN_LABELS, "train.tfrecord")
