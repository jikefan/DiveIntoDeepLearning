# 矩阵
import tensorflow as tf

A = tf.reshape(tf.range(6), (3, 2))

print(A)

# 计算矩阵的转置
At = tf.transpose(A)

print(At)

# 对称矩阵的转置就等于自身
B = tf.constant([
    [1, 2, 3], 
    [2, 0, 4],
    [3, 4, 5]])

print(f"B对称矩阵的转置等于自己吗: {B == tf.transpose(B)}")


C1 = tf.constant([
    [1, 2],
    [4, 5]
])

C2 = tf.constant([
    [1, 2],
    [0, 0]
])

print(C1 == C2)