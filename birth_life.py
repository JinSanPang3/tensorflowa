import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
DATA_FILE = "C:\\Users\\LQ\\Desktop\\tensorflow\\birth_life_2010.txt"
def readdata():
    text = open(DATA_FILE,'r').readlines()[1:]
    data = [line[:-1].split('\t') for line in text]
    births = [float(line[1]) for line in data]
    lifes = [float(line[2]) for line in data]
    data = list(zip(births, lifes))
    n_samples = len(data)
    data = np.asarray(data, dtype=np.float32)
    return data, n_samples

def huber_loss(labels, predictions, delta=4.0):
    residual = tf.abs(labels - predictions)
    def f1(): return 0.5 * tf.square(residual)
    def f2(): return delta * residual - 0.5 * tf.square(delta)
    return tf.cond(residual < delta, f1, f2)

data, n_samples = readdata()
X = tf.placeholder(tf.float32,None,"X")
Y = tf.placeholder(tf.float32,None,"Y")
w = tf.get_variable("W",initializer=tf.constant(0.0))
b = tf.get_variable("bias",initializer=tf.constant(0.0))
Y_predicted = tf.multiply(w,X)+b
loss = tf.square(Y-Y_predicted,"loss")
#loss = huber_loss(Y,Y_predicted)
opt = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
start = time.time()
writer = tf.summary.FileWriter('./g1',tf.get_default_graph())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        total_loss = 0
        for x, y in data:
            _,loss_ = sess.run([opt,loss],feed_dict={X: x,Y: y})
            total_loss+=loss_
    w_out = sess.run(w)
    b_out =  sess.run(b)
    print(w_out)
    print(b_out)
    writer.close()
print('Epoch {0}: {1}'.format(i, total_loss/n_samples))
print('Took: %f seconds' %(time.time() - start))



plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted data')
plt.legend()
plt.show()