import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
from helper import *


# Hyper Parameters
LEARNING_RATE = 1e-4
TAU = 0.001

class ActorNetwork:
    """docstring for ActorNetwork"""
    def __init__(self,sess,state_dim,action_dim,scope):

        self.state_dim = state_dim
        self.action_dim = action_dim
        # create actor network
        self.state_input,self.action_output,self.net = self.create_network(state_dim,action_dim,scope)

        # create target actor network
        self.target_state_input,self.target_action_output,self.target_update,self.target_net = self.create_target_network(state_dim,action_dim,self.net,scope)
        # define training rules
        if scope != 'global/actor':
	    self.q_gradient_input = tf.placeholder("float",[None,self.action_dim])
	    self.parameters_gradients,_ = tf.clip_by_global_norm(tf.gradients(self.action_output,self.net,-self.q_gradient_input),1.0)
	    global_vars_actor = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global/actor')
	    self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,global_vars_actor))
	    sess.run(tf.global_variables_initializer())

        #self.update_target()
        #self.load_network()


    def create_network(self,state_dim,action_dim,scope):
        with tf.variable_scope(scope):

           state_input = tf.placeholder("float",[None,state_dim])
           layer1 = slim.fully_connected(state_input,300,activation_fn=tf.nn.elu,weights_initializer=tf.truncated_normal_initializer(stddev=0.1))
           layer2 = slim.fully_connected(layer1,200,activation_fn=tf.nn.elu,weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
           action_output = slim.fully_connected(layer2,action_dim,activation_fn=tf.sigmoid,weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
           net = [v for v in tf.trainable_variables() if scope in v.name]

           return state_input,action_output,net

    def create_target_network(self,state_dim,action_dim,net,scope):
        state_input,action_output,target_net = self.create_network(state_dim,action_dim,scope+'/target')
        # updating target netowrk
        target_update = []
        for i in range(len(target_net)):
            # theta' <-- tau*theta + (1-tau)*theta'
            target_update.append(target_net[i].assign(tf.add(tf.multiply(TAU,net[i]),tf.multiply((1-TAU),target_net[i]))))
        return state_input,action_output,target_update,target_net

    def update_target(self,sess):
        sess.run(self.target_update)

    def train(self,sess,q_gradient_batch,state_batch):
        sess.run(self.optimizer,feed_dict={
            self.q_gradient_input:q_gradient_batch,
            self.state_input:state_batch
            })

    def actions(self,sess,state_batch):
        return sess.run(self.action_output,feed_dict={
            self.state_input:state_batch
            })

    def action(self,sess,state):
        return sess.run(self.action_output,feed_dict={
            self.state_input:[state]
            })[0]


    def target_actions(self,sess,state_batch):
        return sess.run(self.target_action_output,feed_dict={
            self.target_state_input:state_batch
            })

    # f fan-in size
    def variable(self,shape,f):
        return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))
'''
    def load_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_actor_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"
    def save_network(self,time_step):
        print 'save actor-network...',time_step
        self.saver.save(self.sess, 'saved_actor_networks/' + 'actor-network', global_step = time_step)

'''


