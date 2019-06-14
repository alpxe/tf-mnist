"""
用残差神经网络构建训练模型
"""

# tensorflow __version__ >> 1.12.0
import tensorflow as tf
from tensorflow.python.training import moving_averages
import numpy as np
import cv2

print(tf.__version__)


def _bottleneck(x, d, out, stride=None, scope="bottleneck"):
    """
    三层瓶颈结构为1X1，3X3和1X1卷积层
    其中两个1X1卷积用来减少或增加维度
    3X3卷积可以看作一个更小的输入输出维度的瓶颈
    :param x: 输入
    :param d: 瓶颈值 教程上是out整除4的值
    :param out: 输出的维度值
    :return:
    """

    align = x.get_shape()[-1]  # 获取输入最后的维度 用于维度对齐

    if stride is None:
        stride = 1 if align == out else 2

    with tf.variable_scope(scope):
        # 1x1卷积核 如果stride默认且输入与输出的维度一致，则步长为2 卷积后的size/2
        h = conv2d(x, d, 1, stride=stride, scope="conv_1")  # [batch,size,size,align]->[batch,size,size,d]
        h = batch_norm(h, scope="bn_1")
        h = tf.nn.relu(h)

        # 3x3卷积核 [batch,size,size,d]->[batch,size,size,d]
        h = conv2d(h, d, 3, stride=1, scope="conv_2")
        h = batch_norm(h, scope="bn_2")
        h = tf.nn.relu(h)

        # 1x1卷积核 [batch,size,size,d]->[batch,size,size,out]
        h = conv2d(h, out, 1, stride=1, scope="conv_3")
        h = batch_norm(h, scope="bn_3")

        if align != out:  # 维度不同
            shortcut = conv2d(x, out, 1, stride=stride, scope="conv_4")
            shortcut = batch_norm(shortcut, scope="bn_4")
        else:
            shortcut = x

        return tf.nn.relu(h + shortcut)
    pass


def _block(x, out, n, init_stride=2, scope="block"):
    """
    残差
    :param x: 输入
    :param out: 输出的维度
    :param n: 迭代次数
    :param init_stride: 初始步长值
    :param scope:
    :return:
    """

    with tf.variable_scope(scope):
        bok = out // 4  # 瓶颈值
        net = _bottleneck(x, bok, out, stride=init_stride, scope="bottlencek1")

        for i in range(1, n):
            net = _bottleneck(net, bok, out, scope=("bottlencek%s" % (i + 1)))

        return net
    pass


def batch_norm(x, decay=0.999, epsilon=1e-03, is_training=True,
               scope="scope"):
    x_shape = x.get_shape()
    num_inputs = x_shape[-1]
    reduce_dims = list(range(len(x_shape) - 1))
    with tf.variable_scope(scope):
        beta = tf.get_variable("bate", shape=[num_inputs, ], initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", shape=[num_inputs, ], initializer=tf.constant_initializer(1))

        moving_mean = tf.get_variable("moving_mean", shape=[num_inputs, ], initializer=tf.zeros_initializer(),
                                      trainable=False)
        moving_variance = tf.get_variable("moving_variance", shape=[num_inputs, ], initializer=tf.zeros_initializer(),
                                          trainable=False)

    if is_training:
        mean, variance = tf.nn.moments(x, axes=reduce_dims)
        update_move_mean = moving_averages.assign_moving_average(moving_mean,
                                                                 mean, decay=decay)
        update_move_variance = moving_averages.assign_moving_average(moving_variance,
                                                                     variance, decay=decay)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_move_mean)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_move_variance)
    else:
        mean, variance = moving_mean, moving_variance
    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon)


# 卷积
def conv2d(x, out_size, kernel_size, stride, scope):
    """
    卷积
    :param x: 输入
    :param out_size: 输出的纬度
    :param kernel_size: 卷积核尺寸
    :param stride: 卷积核滑动步长
    :param scope: 空间
    :return:
    """

    align = x.get_shape()[-1]  # (?,28,28,1) 获取张量最后一维的维度，用于维度对齐。*矩阵相乘知识*
    with tf.variable_scope(scope):  # 该命名空间下
        # 创建卷积核
        kernel = tf.get_variable("kernel",
                                 shape=[kernel_size, kernel_size, align, out_size],
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        return tf.nn.conv2d(x, kernel, strides=[1, stride, stride, 1], padding="SAME")


def max_pool(x, pool_size, stride, scope):
    """
    池化
    :param x: 输入的张量
    :param pool_size: 池化核尺寸
    :param stride: 池化核步长，会影响维度
    :param scope:
    :return:
    """
    with tf.variable_scope(scope):
        return tf.nn.max_pool(x, [1, pool_size, pool_size, 1], strides=[1, stride, stride, 1], padding="SAME")


def avg_pool(x, pool_size, scope):
    with tf.variable_scope(scope):
        return tf.nn.avg_pool(x, [1, pool_size, pool_size, 1],
                              strides=[1, pool_size, pool_size, 1], padding="VALID")


def __format(record):
    fs = {
        "label": tf.FixedLenFeature([1], dtype=tf.int64),
        "image": tf.FixedLenFeature([28 * 28], dtype=tf.int64)
    }
    fats = tf.parse_single_example(record, features=fs)

    # label
    label = tf.reshape(fats["label"], [])
    label = tf.one_hot(label, 10)
    label = tf.cast(label, dtype=tf.float32)

    # image
    image = tf.reshape(fats["image"], [28, 28, 1])
    image = tf.cast(image, dtype=tf.float32)

    return label, image


# 1.>>>>得到数据
dataset = tf.data.TFRecordDataset("tfrecords/train.tfrecords")
dataset = dataset.repeat()
dataset = dataset.map(__format)

dataset = dataset.batch(50)

iterator = dataset.make_one_shot_iterator()
label, image = iterator.get_next()

# 2.>>>>残差神经网络
with tf.variable_scope("resnet"):
    net = conv2d(image, 32, 3, 1, "conv1")  # 卷积 [?,28,28,1] -> [?,28,28,32]
    net = tf.nn.relu(batch_norm(net, scope="bn1"))  # BN >>> relu
    net = max_pool(net, 2, 2, "maxpool1")  # [?,28,28,32] -> [?,14,14,32]

    net = _block(net, 256, 3, 1, scope="block_2")  # [?,14,14,256]
    print(net.get_shape())
    net = _block(net, 512, 4, 1, scope="block3")  # [?,14,14,512]
    print(net.get_shape())
    net = _block(net, 1024, 3, 2, scope="block4")  # [?,7,7,1024]
    print(net.get_shape())

    # 展开
    net = avg_pool(net, 7, scope="avgpool5")  # [?,1,1,1024]
    print(net.get_shape())

    # 挤压维度
    # axis：用于指定需要删除的维度，但是指定的维度必须为单维度，否则将会报错，若axis为空，则删除所有单维度的条目
    net = tf.squeeze(net, [1, 2], name="SpatialSqueeze")  # -> [batch, 2048]
    print(net.get_shape())

with tf.variable_scope("logit"):
    align = net.get_shape()[-1]
    weight = tf.get_variable("weight", shape=[align, 10], dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.get_variable("bias", shape=[10], dtype=tf.float32,
                           initializer=tf.zeros_initializer())

    logit = tf.nn.xw_plus_b(net, weight, bias)

    softmax = tf.clip_by_value(tf.nn.softmax(logit), 1e-10, 1.0, name="softmax")  # 值保护不会出现0和大于1

    loss = tf.reduce_mean(-tf.reduce_sum(label * (tf.log(softmax) / tf.log(2.)), 1))
    tf.summary.scalar('loss', loss)  # 与tensorboard 有关

    train = tf.train.AdamOptimizer(1e-4).minimize(loss)

    pass

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    merge = tf.summary.merge_all()  # tensorboard 相关
    writer = tf.summary.FileWriter("graph/", graph=sess.graph)  # 写入日志log

    # saver = tf.train.Saver()  # 初始化 Saver

    for i in range(5000 + 1):
        sess.run([train, loss])
        if i % 10 == 0:
            _, ls, sx, mrg = sess.run([train, loss, softmax, merge])
            writer.add_summary(mrg, i)  #
            print("[step_{0}]:\n\tloss损失值:{1}\n\ttsoftmax:{2}\n".format(i, ls, sx))

            # saver.save(sess, save_path='ckp/')  # 储存神经网络的变量
            pass
        pass
    pass

    writer.close()

    # 训练完毕 保存模型文件
    builder = tf.saved_model.builder.SavedModelBuilder("model_data/")
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING])
    builder.save()
