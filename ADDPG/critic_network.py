import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
from helper import *


LEARNING_RATE = 1e-4
TAU = 1e-4
L2 = 0.01

class CriticNetwork:
    """docstring for CriticNetwork"""
    def __init__(self,sess,state_dim,action_dim,scope):
        self.time_step = 0
        # create q network
        self.state_input,\
        self.action_input,\
        self.q_value_output,\
        self.net = self.create_q_network(state_dim,action_dim,True,scope)

        # create target q network (the same structure with q network)
        if scope == 'worker_1/critic':
            self.target_state_input,self.target_action_input,self.target_q_value_output,self.target_update = self.create_target_q_network(state_dim,action_dim,True,self.net,scope)

        if scope == 'worker_1/critic':
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope)
    	    with tf.control_dependencies(update_ops):
                self.y_input = tf.placeholder("float",[None,1])
                weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.net])
                
                self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output)) + weight_decay
                self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
                self.parameters_gradients,_ = zip(*self.optimizer.compute_gradients(self.cost,self.net))
                self.parameters_graidents,self.global_norm = tf.clip_by_global_norm(self.parameters_gradients,1.0)
                self.optimizer = self.optimizer.apply_gradients(zip(self.parameters_gradients,self.net))
                self.action_gradients = tf.gradients(self.q_value_output,self.action_input)

        sess.run(tf.global_variables_initializer())

    def create_q_network(self,state_dim,action_dim,phase,scope):
        with tf.variable_scope(scope):

            state_input = tf.placeholder("float",[None,state_dim])
            action_input = tf.placeholder("float",[None,action_dim])

	    h1 = dense_relu_batch(state_input,128,phase)
	    h1_a = dense_relu_batch(action_input,128,phase)
	    h2 = dense(tf.add(h1,h1_a),128,tf.nn.relu,tf.contrib.layers.xavier_initializer())
	    q_value_output = dense(h2,1,None,tf.random_uniform_initializer(-3e-3,3e-3))
            net = [v for v in tf.trainable_variables() if scope in v.name]

            return state_input,action_input,q_value_output,net

    def create_target_q_network(self,state_dim,action_dim,phase,net,scope):
        state_input,action_input,q_value_output,target_net = self.create_q_network(state_dim,action_dim,phase,scope+'/target') 
        target_update = []
        ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]
        '''
        for i in range(len(target_net)):
            # theta' <-- tau*theta + (1-tau)*theta'
            target_update.append(target_net[i].assign(tf.add(tf.multiply(TAU,net[i]),tf.multiply((1-TAU),target_net[i]))))
            '''
        return state_input,action_input,q_value_output,target_update

    def update_target(self,sess):
        sess.run(self.target_update)

    def train(self,sess,y_batch,state_batch,action_batch):
        self.time_step += 1
        return sess.run([self.optimizer,self.cost,self.y_input,self.q_value_output,self.global_norm],feed_dict={
            self.y_input:y_batch,
            self.state_input:state_batch,
            self.action_input:action_batch,
            self.target_state_input:state_batch,
            self.target_action_input:action_batch
            })

    def gradients(self,sess,state_batch,action_batch):
        return sess.run(self.action_gradients,feed_dict={
            self.state_input:state_batch,
            self.action_input:action_batch,
            self.target_state_input:state_batch,
            self.target_action_input:action_batch
            })[0]

    def target_q(self,sess,state_batch,action_batch):
        return sess.run(self.target_q_value_output,feed_dict={
            self.target_state_input:state_batch,
            self.target_action_input:action_batch
            })

    def q_value(self,sess,state_batch,action_batch):
        return sess.run(self.q_value_output,feed_dict={
            self.state_input:state_batch,
            self.action_input:action_batch
            })

    # f fan-in size
    def variable(self,shape,f):
        return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))
'''
    def load_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_critic_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"

    def save_network(self,time_step):
        print 'save critic-network...',time_step
        self.saver.save(self.sess, 'saved_critic_networks/' + 'critic-network', global_step = time_step)
'''
