from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import time
MNIST = input_data.read_data_sets('data/mnist', one_hot=True)
batch_size = 128

print(MNIST.train.num_examples/batch_size)


X = tf.placeholder(tf.float32,[batch_size,784],name='X_placeholder')## 28*28
Y = tf.placeholder(tf.float32,[batch_size,10],name='Y_placeholder') ##0-9

wei = tf.get_variable('weights',initializer=tf.random_normal(shape=[784,10],stddev=0.01))
b = tf.get_variable('bias',initializer=tf.zeros([1,10]))

logit = tf.matmul(X, wei)+b

en = tf.nn.softmax_cross_entropy_with_logits(logits=logit,labels=Y,name='loss')
#en = tf.square(logit-Y)
loss = tf.reduce_mean(en)

op = tf.train.AdamOptimizer(0.001).minimize(loss)
writer = tf.summary.FileWriter('/g2',tf.get_default_graph())
with tf.Session() as sess :
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    n_batch = int(MNIST.train.num_examples/batch_size)
    for i in range(30):
        total_loss = 0
        for _ in range(n_batch):
            X_batch,Y_batch = MNIST.train.next_batch(batch_size)
            _,loss_batch = sess.run([op,loss],feed_dict={X:X_batch,Y:Y_batch})
            total_loss +=loss_batch
        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batch))
    print('Total time: {0} seconds'.format(time.time() - start_time))

    preds = tf.nn.softmax(logit)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))  # need numpy.count_nonzero(boolarr) :(

    n_batches = int(MNIST.test.num_examples / batch_size)
    total_correct_preds = 0

    '''a,b = MNIST.test.next_batch(1)
    print(a)'''
    for i in range(n_batches):
        X_batch, Y_batch = MNIST.test.next_batch(batch_size)
        accuracy_batch = sess.run([accuracy], feed_dict={X: X_batch, Y: Y_batch})
        total_correct_preds += accuracy_batch[0]

    print('Accuracy {0}'.format(total_correct_preds / MNIST.test.num_examples))

writer.close()