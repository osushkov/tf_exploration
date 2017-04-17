
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def createSampleData(samples):
    a = 10.0;
    b = -5.0;
    noise_sd = 10.0

    xs = np.random.rand(1, samples) * 100.0
    ys = xs * a + b + np.random.normal(0.0, noise_sd, samples)

    return xs, ys


# plt.scatter(xs, ys)
# plt.show()

batch_size = 100

graph = tf.Graph()
with graph.as_default():
    av = tf.Variable(np.random.uniform(-100.0, 100.0, 1).reshape(1, 1), dtype=tf.float32)
    bv = tf.Variable(np.random.uniform(-100.0, 100.0, 1).reshape(1, 1), dtype=tf.float32)

    xv = tf.placeholder(tf.float32, shape=(1, batch_size))
    yv = tf.matmul(av, xv) + bv

    print yv.get_shape()
