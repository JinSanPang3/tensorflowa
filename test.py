import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
c = tf.multiply(a,b)
x = tf.add(c, b)

writer = tf.summary.FileWriter('./g',tf.get_default_graph())
with tf.Session()as sess:
    print(sess.run(x))
writer.close()
range(1,5,1)
#tensorboard --logdir="./g" --port 6006