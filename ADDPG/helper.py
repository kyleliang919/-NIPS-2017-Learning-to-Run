import numpy as np
import tensorflow as tf


# Helper Function------------------------------------------------------------------------------------------------------------
# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def update_target_network(network,target_network,TAU):
    network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, network)
    target_network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, target_network)

    target_update = []
    for network_var,target_network_var in zip(network_vars,target_network_vars):
        target_update.append(target_network_var.assign(tf.add(tf.multiply(TAU,network_var),tf.multiply((1-TAU),target_network_var))))
    return target_update

# Normalize state 
def process_frame(s):
    s = np.asarray(s)
    s = (s-np.mean(s)) / np.std(s)
    return s

# process state (the last 3 entires are obstacle info which should not be processed)
def process_state(s,s1):
    s = np.asarray(s)
    s1 = np.asarray(s1)
    s = np.hstack((s1[:-3]-s[:-3],s[-3:]))
    return s
    
def engineered_action(seed):
    a = np.ones(18)*0.05
    if seed < .5:
        a[17]=0.9
        a[0]=0.9
        a[3]=0.9
        a[4]=0.9
        a[8]=0.9
        a[11]=0.9
        a[12]=0.9
        a[13]=0.9
        a[10]=0.9
    else:
        a[8]=0.9
        a[9]=0.9
        a[12]=0.9
        a[13]=0.9
        a[17]=0.9
        a[2]=0.9
        a[3]=0.9
        a[4]=0.9
        a[1]=0.9 
    return a

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

    
#These functions allows us to update the parameters of our target network with those of the primary network.
def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars/2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)
    total_vars = len(tf.trainable_variables())
    a = tf.trainable_variables()[0].eval(session=sess)
    b = tf.trainable_variables()[total_vars/2].eval(session=sess)
    if a.all() == b.all():
        print("Target Set Success")
    else:
        print("Target Set Failed")
        

    



