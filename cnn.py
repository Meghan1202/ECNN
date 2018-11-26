import tensorflow
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

classes = 10
batch_size = 128

x = tensorflow.placeholder(tensorflow.float32, [None, 784])
y = tensorflow.placeholder(tensorflow.float32, [None, 10])


def custom_pool(x):
    return (tensorflow.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME') + tensorflow.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME'))/2


def cnn(x):

     weights = {
                'conv_l1': tensorflow.Variable(tensorflow.random_normal([5, 5, 1, 32])),
                'conv_l2': tensorflow.Variable(tensorflow.random_normal([5, 5, 32, 64])),
                'fully_connected': tensorflow.Variable(tensorflow.random_normal([7*7*64, 1024])),
                'output_l': tensorflow.Variable(tensorflow.random_normal([1024, classes])),
     }

     biases = {
                'conv_l1': tensorflow.Variable(tensorflow.random_normal([32])),
                'conv_l2': tensorflow.Variable(tensorflow.random_normal([64])),
                'fully_connected': tensorflow.Variable(tensorflow.random_normal([1024])),
                'output_l': tensorflow.Variable(tensorflow.random_normal([classes])),
     }

     x = tensorflow.reshape(x, shape=[-1, 28, 28, 1])


     conv1 = tensorflow.nn.relu(tensorflow.add(tensorflow.nn.conv2d(x, weights['conv_l1'], strides=[1, 1, 1, 1], padding='SAME'),
                                               biases['conv_l1']))
     pool1 = custom_pool(conv1)

     conv2 = tensorflow.nn.relu(
         tensorflow.add(tensorflow.nn.conv2d(pool1, weights['conv_l2'], strides=[1, 1, 1, 1], padding='SAME'),
                        biases['conv_l2']))

     pool2 = custom_pool(conv2)


     fully_connected = tensorflow.reshape(pool2, [-1, 7*7*64])

     fully_connected = tensorflow.nn.relu(tensorflow.matmul(fully_connected,weights['fully_connected'])
                                          + biases['fully_connected'])
    
     tensorflow.nn.dropout(fully_connected, 0.8)

     output = tensorflow.matmul(fully_connected, weights['output_l']) + biases['output_l']

     return output


def train_neural_network(x):

    prediction = cnn(x)
    cost = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tensorflow.train.AdamOptimizer().minimize(cost)
    hm_epochs = 10
    saver = tensorflow.train.Saver()
    with tensorflow.Session() as sess:
        sess.run(tensorflow.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tensorflow.equal(tensorflow.argmax(prediction, 1), tensorflow.argmax(y, 1))

        accuracy = tensorflow.reduce_mean(tensorflow.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        saver.save(sess, "customPool.ckpt")

train_neural_network(x)
