
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def createSampleData(samples):
    a = 10.0;
    b = -5.0;
    noise_sd = 0.1

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
    av = tf.Variable(np.random.uniform(-1.0, 1.0, 1).reshape(1, 1), dtype=tf.float32)
    bv = tf.Variable(np.random.uniform(-1.0, 1.0, 1).reshape(1, 1), dtype=tf.float32)

    xv = tf.placeholder(tf.float32, shape=(1, batch_size))
    yv = tf.placeholder(tf.float32, shape=(1, batch_size))
    ypred = tf.add(tf.matmul(av, xv), bv)

    loss = tf.reduce_mean(tf.squared_difference(ypred, yv))
    loss_summary = tf.summary.scalar("loss", loss)
    opt = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    file_writer = tf.summary.FileWriter('logs', sess.graph)

    data_x, data_y = createSampleData(10000)
    for i in range(1000):
        batch_x, batch_y = makeBatch(batch_size, data_x, data_y)
        _, l, a, b, ls = sess.run([opt, loss, av, bv, loss_summary], feed_dict={xv: batch_x, yv: batch_y})
        file_writer.add_summary(ls, global_step=i)
        print("iter: " + str(i) + " loss: " + str(l))
    print(a, b)
