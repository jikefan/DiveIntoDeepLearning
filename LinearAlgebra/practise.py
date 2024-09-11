# 练习

# 证明一个矩阵的转置的转置是A
import tensorflow as tf

A = tf.constant([
    [1, 2, 3],
    [4, 5, 6]
])

print(A == tf.transpose(tf.transpose(A)))


# 给出两个矩阵和，证明“它们转置的和”等于“它们和的转置”
B = tf.constant([
    [6, 7, 8],
    [1, 2, 3]
])

print(tf.transpose(A + B) == tf.transpose(A) + tf.transpose(B))

# 本节中定义了形状的张量X。len(X)的输出结果是什么？
T = tf.reshape(tf.range(24), (2, 3, 4))
print(len(T))

# 运行A/A.sum(axis=1)，看看会发生什么。请分析一下原因？
print(A / tf.reduce_sum(A))

