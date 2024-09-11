# 张量

from calendar import c
import tensorflow as tf

X = tf.reshape(tf.range(24), (2, 3, 4))

print(X)

A = tf.reshape(tf.range(20, dtype=tf.float32), (5,4))

B = A

print(A)

print(A + B)

# 让张量中的对应的元素相乘
print(A * B)

# 降维
x = tf.range(4, dtype=tf.float32)

print(tf.reduce_sum(x))

# 可以计算任意维度的张量元素和
print(tf.reduce_sum(A))

print(tf.reduce_sum(A, axis=1))

#tf.reduce_sum(A, axis=[0, 1])  # 结果和tf.reduce_sum(A)相同

# 计算平均值
print(tf.reduce_mean(A))

# 沿着指定轴降低张量维度，计算平均值
print(tf.reduce_mean(A, axis=1))

# 非降维求和
sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
print(sum_A)

print(A / sum_A)

print(tf.cumsum(A, axis=0))