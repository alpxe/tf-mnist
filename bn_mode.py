import tensorflow as tf
import numpy as np

# batch norm layer
def batch_norm(x, decay=0.999, epsilon=1e-03, is_training=True,
               scope="scope"):
    x_shape = x.get_shape()
    num_inputs = x_shape[-1]
    reduce_dims = list(range(len(x_shape) - 1))  # 特征的通道数
    with tf.variable_scope(scope):
        # 声明BN中唯一需要学习的两个参数，y=gamma*x+beta
        beta = tf.get_variable("bate", shape=[num_inputs, ], initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", shape=[num_inputs, ], initializer=tf.constant_initializer(1))

        # 计算当前整个batch的均值与方差
        batch_mean, batch_var = tf.nn.moments(x, axes=reduce_dims, name="moments")

        # 采用滑动平均更新均值与方差
        ema = tf.train.ExponentialMovingAverage(decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # 训练时，更新均值与方差，测试时使用之前最后一次保存的均值与方差
        mean, var = tf.cond(tf.equal(is_training, True), mean_var_with_update, lambda: (ema.average(batch_mean),ema.average(batch_var)))
        return tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)
