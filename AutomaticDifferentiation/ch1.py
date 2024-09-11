# 自动微分简单例子

import tensorflow as tf

x = tf.range(4, dtype=tf.float32)

x = tf.Variable(x)

with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
    # x_grad是y对x的导数
    x_grad = t.gradient(y, x)

    print(y)
    print(x_grad)
