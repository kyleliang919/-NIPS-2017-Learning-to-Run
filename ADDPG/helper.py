import numpy as np
import tensorflow as tf


# Helper Function------------------------------------------------------------------------------------------------------------
# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def lrelu(x, alpha=0.2):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def update_graph(from_vars,to_vars):
    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


# Normalize state 
def normalize(s):
    s = np.asarray(s)
    s = (s-np.mean(s)) / np.std(s)
    return s

# process state (the last 3 entires are obstacle info which should not be processed)
def process_state(s,s1):
    s = np.asarray(s)
    s1 = np.asarray(s1)
    s_14 = (s1[22:36]-s[22:36]) / 0.01
    s_3 = (s1[38:]-s[38:]) / 0.01
    s = np.hstack((s1[:36],s_14,s1[36:],s_3))
    
    # transform into all relative quantities
    x_pos = [1,22,24,26,28,30,32,34]
    y_pos = [i+1 for i in x_pos]
    for i in x_pos:
        s[i] -= s[18]
    for j in y_pos:
        s[j] -= s[19]
    
    x_vs = [i+14 for i in x_pos]
    x_vs[0] = 4
    y_vs = [i+1 for i in x_vs]
    for i in x_vs:
        s[i] -= s[20]
    for j in y_vs:
        s[j] -= s[21]
    # transform cm as origin
    s[18:22] = 0
        
    return s
    
def engineered_action(seed):
    test = np.ones(18)*0.05
    if seed < 0.5:
        test[0] = 0.3
        test[3] = 0.8
        test[4] = 0.5
        test[6] = 0.3
        test[8] = 0.8
        test[9] = 0.3
        test[11] = 0.5
        test[14] = 0.3
        test[17] = 0.5
    else:
        test[9] = 0.3
        test[12] = 0.8
        test[13] = 0.5
        test[15] = 0.3
        test[17] = 0.8
        test[0] = 0.3
        test[2] = 0.5
        test[3] = 0.3
        test[8] = 0.5
            
    return test

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def dense(x, size, activation, weights_initializer): # dense layer 
    return tf.contrib.layers.fully_connected(x, size, 
                                             activation_fn=activation,
                                             weights_initializer=weights_initializer)

def dense_relu_batch(x, size, phase): # batch_normalize dense layer
    h1 = tf.contrib.layers.fully_connected(x, size, 
                                               activation_fn=tf.nn.relu)
    h2 = tf.contrib.layers.batch_norm(h1,center=True, scale=True, 
                                          is_training=phase)
    return h2
        
def n_step_transition(episode_buffer,n_step,gamma):
    _,_,_,s1,done = episode_buffer[-1]
    s,action,_,_,_ = episode_buffer[-1-n_step]
    r = 0
    for i in range(n_step):
      r += episode_buffer[-1-n_step+i][2]*gamma**i
    return [s,action,r,s1,done]


