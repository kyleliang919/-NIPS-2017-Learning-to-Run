import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, [batch_size, 10, 16])

with tf.Session() as sess:
    mu = tf.placeholder(shape=[3],dtype=tf.float32)
    var = tf.placeholder(shape=[3],dtype=tf.float32)
    normal = tf.distributions.Normal(mu,tf.sqrt(var))
    act = tf.placeholder(shape=[3],dtype=tf.float32)
    log_prob = normal.log_prob(act)
    entropy = normal.entropy()
    feed_dict={mu:np.zeros(3),var:np.ones(3)*20,act:np.array([0,-0.5,1])}
    print sess.run([log_prob,entropy],feed_dict=feed_dict)
