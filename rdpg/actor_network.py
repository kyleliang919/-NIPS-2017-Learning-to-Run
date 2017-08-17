import tensorflow as tf
import numpy as np
import math
from tensorflow.contrib import rnn

# Hyper Parameters
LSTM_HIDDEN_UNIT = 300
LEARNING_RATE = 1e-4
TAU = 0.001
BATCH_SIZE = 64

class ActorNetwork:
    """docstring for ActorNetwork"""
    def __init__(self,sess,state_dim,action_dim):

        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        # create actor network

        self.state_input,self.action_output,self.net = self.create_network(state_dim,action_dim,"beh")

        # create target actor network
        self.target_state_input,self.target_action_output,self.target_update,self.target_net = self.create_target_network(state_dim,action_dim,self.net)

        # define training rules
        self.create_training_method()

        self.sess.run(tf.initialize_all_variables())

        self.update_target()
        #self.load_network()

    def create_training_method(self):
        self.q_gradient_input = tf.placeholder("float",[None,self.action_dim])
        self.parameters_gradients = tf.gradients(self.action_output,self.net,-self.q_gradient_input)
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,self.net))

    def create_network(self,state_dim,action_dim,scope):
        with tf.variable_scope(scope,reuse=False) as s:

            state_input = tf.placeholder("float",[None,None,state_dim])

            # creating the recurrent part
            lstm_cell=rnn.BasicLSTMCell(LSTM_HIDDEN_UNIT)
            lstm_output,lstm_state=tf.nn.dynamic_rnn(cell=lstm_cell,inputs=state_input,dtype=tf.float32)
            W3 = tf.Variable(tf.random_uniform([lstm_cell.state_size,action_dim],-3e-3,3e-3))
            b3 = tf.Variable(tf.random_uniform([action_dim],-3e-3,3e-3))

            action_output = tf.tanh(tf.matmul(lstm_state,W3) + b3)

            net = [v for v in tf.trainable_variables() if scope in v.name]

        return state_input,action_output,net

    def create_target_network(self,state_dim,action_dim,net):
        state_input,action_output,target_net = self.create_network(state_dim,action_dim,"tare")
        # updating target netowrk
        target_update = []
        for i in len(target_net):
            # theta' <-- tau*theta + (1-tau)*theta'
            target_update.append(target_net[i].assign(tf.add(tf.multiply(TAU,net[i]),tf.multiply((1-TAU),target[i]))))
        return state_input,action_output,target_update,target_net

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self,q_gradient_batch,state_batch):
        self.sess.run(self.optimizer,feed_dict={
            self.q_gradient_input:q_gradient_batch,
            self.state_input:state_batch
            })

    def actions(self,state_batch):
        return self.sess.run(self.action_output,feed_dict={
            self.state_input:state_batch
            })

    def action(self,state_batch):
        return self.sess.run(self.action_output,feed_dict={
            self.state_input:state_batch
            })[0]


    def target_action(self,state_batch):
        return self.sess.run(self.target_action_output,feed_dict={
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


