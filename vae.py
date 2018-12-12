import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import os

mnist = input_data.read_data_sets('/tmp/', one_hot=True)

latent_dim = 10

X = tf.placeholder(tf.float32, [None, 28, 28])
Y = tf.placeholder(tf.float32, [None, 28, 28])
Y_ = tf.reshape(Y, shape=[-1, 28 * 28])


def conv(x, nf):
    return tf.layers.conv2d(x, filters=nf, kernel_size=4, strides=1, padding='SAME', activation=tf.nn.relu)


def deconv(x, nf):
    return tf.layers.conv2d_transpose(x, filters=nf, kernel_size=4, strides=1, padding='SAME', activation=tf.nn.relu)


def encoder(X):
    x = tf.reshape(X, [-1, 28, 28, 1])
    x = conv(x, 64)
    x = tf.layers.batch_normalization(x)
    x = conv(x, 64)
    x = tf.layers.batch_normalization(x)
    x = conv(x, 128)
    x = tf.layers.batch_normalization(x)
    x = tf.layers.flatten(x)
    mn = tf.layers.dense(x, units=latent_dim)
    sd = 0.5 * tf.layers.dense(x, units=latent_dim)
    epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], latent_dim]))
    z = mn + tf.multiply(epsilon, tf.exp(sd))
    return z, mn, sd


def decoder(z):
    x = tf.layers.dense(z, units=49 / 2, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=(49 / 2) * 2, activation=tf.nn.relu)
    x = tf.reshape(x, [-1, 7, 7, 1])
    x = deconv(x, 64)
    x = tf.layers.batch_normalization(x)
    x = deconv(x, 64)
    x = tf.layers.batch_normalization(x)
    x = deconv(x, 128)
    x = tf.layers.batch_normalization(x)
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, units=28*28, activation=tf.nn.sigmoid)
    img = tf.reshape(x, shape=[-1, 28, 28])
    return img


s, mn, sd = encoder(X)
dec = decoder(s)

unreshaped = tf.reshape(dec, [-1, 28*28])
img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_), 1)
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
loss = tf.reduce_mean(img_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

for i in range(30000):
    batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=128)[0]]
    sess.run(optimizer, feed_dict={X: batch, Y: batch})
    if not i % 200:
        ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mn, sd],
                                               feed_dict={X: batch, Y: batch})
        plt.imshow(np.reshape(batch[0], [28, 28]), cmap='gray')
        plt.show()
        plt.imshow(d[0], cmap='gray')
        plt.show()
        saver.save(sess, os.path.join(os.getcwd(), f'/model{i}.ckpt'))
        print(i, ls, np.mean(i_ls), np.mean(d_ls))




