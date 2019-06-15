import tensorflow as tf
import os
import gzip
import struct
import numpy as np


iamge_path = os.path.join("MNIST_DATA", 'train-images-idx3-ubyte.gz')
with open(iamge_path, 'rb') as f:  # 打开文件
    with gzip.GzipFile(fileobj=f)as bytestream:  # 解压缩
        img_buf = bytestream.read()

label_path = os.path.join("MNIST_DATA", 'train-labels-idx1-ubyte.gz')
with open(label_path, 'rb') as f:
    with gzip.GzipFile(fileobj=f)as bytestream:
        label_buf = bytestream.read()

# 获取字节流中，前4个字符。<>,<样本数量>,<样本纬度>,<样本纬度>
magic, items, row, col = struct.unpack_from(">IIII", img_buf, 0)

img_hd = struct.calcsize(">IIII")
label_hd = struct.calcsize(">II")

index=5

resimg = struct.unpack_from(">{0}B".format(row * col), img_buf, img_hd + index * row * col)
label, = struct.unpack_from(">B", label_buf, label_hd + index * 1)

resimg=np.array(resimg)
resimg=np.reshape(resimg,[1,28,28,1])
# print(resimg)
print("正确的数字是:{0}".format(label))

with tf.Session(graph=tf.Graph()) as sess:
    # 读取完成的模型
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], "model_data/")
    # # 输入
    is_training = sess.graph.get_tensor_by_name("holder/is_training:0")

    img = sess.graph.get_tensor_by_name("holder/input_image:0")

    softmax = sess.graph.get_tensor_by_name("logit/softmax_1:0")

    _res = sess.run(softmax, feed_dict={img: resimg, is_training: False})


    print(_res)
    arr=_res[0].tolist()
    print(np.argmax(arr))
