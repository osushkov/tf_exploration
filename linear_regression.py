
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def createSampleData(samples):
    a = 10.0;
    b = -5.0;
    noise_sd = 1.0

    xs = np.random.rand(1, samples) * 1.0
    ys = xs * a + b + np.random.normal(0.0, noise_sd, samples)

    return xs, ys


def makeBatch(batch_size, data_x, data_y):
    indices = np.random.permutation(data_x.shape[1])[:batch_size]
    return data_x[:,indices], data_y[:,indices]


# plt.scatter(xs, ys)
# plt.show()

batch_size = 1000

graph = tf.Graph()
with graph.as_default():
    av = tf.Variable(np.random.uniform(-10.0, 10.0, 1).reshape(1, 1), dtype=tf.float32)
    bv = tf.Variable(np.random.uniform(-10.0, 10.0, 1).reshape(1, 1), dtype=tf.float32)

    xv = tf.placeholder(tf.float32, shape=(1, batch_size))
    yv = tf.placeholder(tf.float32, shape=(1, batch_size))
    ypred = tf.add(tf.matmul(av, xv), bv)

    loss = tf.reduce_mean(tf.squared_difference(ypred, yv))
    opt = tf.train.AdamOptimizer().minimize(loss)

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    data_x, data_y = createSampleData(10000)
    for i in range(10000):
        batch_x, batch_y = makeBatch(batch_size, data_x, data_y)
        _, l, a, b = sess.run([opt, loss, av, bv], feed_dict={xv: batch_x, yv: batch_y})
        print("iter: " + str(i) + " loss: " + str(l))
    print(a, b)
