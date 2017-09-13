import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
from helper import *


# Hyper Parameters
LEARNING_RATE = 5e-5
TAU = 1e-4

class ActorNetwork:
    """docstring for ActorNetwork"""
    def __init__(self,sess,state_dim,action_dim,scope):

        self.state_dim = state_dim
        self.action_dim = action_dim
        
        with tf.variable_scope(scope):
            self.phase = tf.placeholder("bool")

        # create actor network
        if scope == 'worker_1/actor':
            self.state_input,self.action_output,self.net = self.create_network(state_dim,action_dim,self.phase,scope)
        else: # for the rest workers & global, training phase == False
            self.state_input,self.action_output,self.net = self.create_network(state_dim,action_dim,False,scope)           

        # create target actor network
        if scope == 'worker_1/actor':
            self.target_state_input,self.target_action_output,self.target_update,self.target_net = self.create_target_network(state_dim,action_dim,True,self.net,scope)
        # define training rules
        if scope == 'worker_1/actor':
	    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope)
    	    with tf.control_dependencies(update_ops):
	        self.q_gradient_input = tf.placeholder("float",[None,self.action_dim])
	        self.parameters_gradients,self.global_norm = tf.clip_by_global_norm(tf.gradients(self.action_output,self.net,self.q_gradient_input),1.0)
		self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,self.net))
	sess.run(tf.global_variables_initializer())

        #self.update_target()
        #self.load_network()


    def create_network(self,state_dim,action_dim,phase,scope):
        with tf.variable_scope(scope):

           state_input = tf.placeholder("float",[None,state_dim])
	   h1 = dense_relu_batch(state_input,128,phase)
	   h2 = dense_relu_batch(h1,128,phase)
	   action_output = dense(h2,action_dim,tf.tanh,tf.random_uniform_initializer(-3e-3,3e-3))
           net = [v for v in tf.trainable_variables() if scope in v.name]

           return state_input,action_output,net

    def create_target_network(self,state_dim,action_dim,phase,net,scope):
        state_input,action_output,target_net = self.create_network(state_dim,action_dim,phase,scope+'/target')
        # updating target netowrk
        target_update = []
        ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]
        '''
        for i in range(len(target_net)):
            # theta' <-- tau*theta + (1-tau)*theta'
            target_update.append(target_net[i].assign(tf.add(tf.multiply(TAU,net[i]),tf.multiply((1-TAU),target_net[i]))))
            '''
        return state_input,action_output,target_update,target_net

    def update_target(self,sess):
        sess.run(self.target_update)

    def train(self,sess,q_gradient_batch,state_batch):
        return sess.run([self.optimizer,self.global_norm],feed_dict={
            self.q_gradient_input:q_gradient_batch,
            self.state_input:state_batch,
            self.target_state_input:state_batch,
	    self.phase:True
            })

    def actions(self,sess,state_batch):
        return sess.run(self.action_output,feed_dict={
            self.state_input:state_batch,
	    self.phase:True
            })

    def action(self,sess,state):
        return sess.run(self.action_output,feed_dict={
            self.state_input:[state],
	    self.phase:False
            })[0]


    def target_actions(self,sess,state_batch):
        return sess.run(self.target_action_output,feed_dict={
            self.target_state_input:state_batch,
            self.phase:True
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


