# 点积，相同位置的按元素乘积的和
import tensorflow as tf

# tf.Tensor([0. 1. 2. 3.], shape=(4,), dtype=float32)
x = tf.range(4, dtype=tf.float32)
# tf.Tensor([1. 1. 1. 1.], shape=(4,), dtype=float32)
y = tf.ones(4, dtype=tf.float32)
print(x, y)

print(tf.tensordot(x, y, axes=1))

print(tf.reduce_sum(x * y))

A = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))

print(A)
# A的列数必须和x的行数一样才能计算向量积，应用个人觉得就是乘上权重向量累和
print(tf.linalg.matvec(A, x))

B = tf.ones((4,3), tf.float32)

print(B)

# 矩阵乘法
print(tf.matmul(A, B))

# 范数，L2范数

u = tf.constant([3.0, -4.0])

print(tf.norm(u))

# L1范数，与L2范数相比，L1范数受异常值的影响较小
l1 = tf.reduce_sum(tf.abs(u))
print(l1)

# 矩阵范数，所有元素的平方和再开根号
print(tf.norm(tf.ones((4, 9))))