import tensorflow as tf
import numpy as np
import math
from tensorflow.contrib import rnn

# LAYER1_SIZE = 400
# LAYER2_SIZE = 300
LSTM_HIDDEN_UNIT = 300
LEARNING_RATE = 1e-3
TAU = 0.001
L2 = 0.01

class CriticNetwork:
    """docstring for CriticNetwork"""
    def __init__(self,sess,state_dim,action_dim):
        self.time_step = 0
        self.sess = sess
        # create q network
        self.state_input,\
        self.action_input,\
        self.q_value_output,\
        self.net = self.create_q_network(state_dim,action_dim,"cbeh")

        # create target q network (the same structure with q network)
        self.target_state_input,\
        self.target_action_input,\
        self.target_q_value_output,\
        self.target_update = self.create_target_q_network(state_dim,action_dim,self.net,"ctare")

        self.create_training_method()

        # initialization
        self.sess.run(tf.initialize_all_variables())

        self.update_target()

    def create_training_method(self):
        # Define training optimizer
        self.y_input = tf.placeholder("float",[None,1])
        weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.net])
        self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output)) + weight_decay
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
        self.action_gradients = tf.gradients(self.q_value_output,self.action_input)

    def create_q_network(self,state_dim,action_dim,scope):
        # the layer size could be changed
        with tf.variable_scope(scope,reuse=False) as s:
            state_input = tf.placeholder("float",[None,None,state_dim])
            action_input = tf.placeholder("float",[None,None,action_dim])

            # creating the recurrent part
            lstm_cell=rnn.BasicLSTMCell(LSTM_HIDDEN_UNIT)
            lstm_output,lstm_state=tf.nn.dynamic_rnn(cell=lstm_cell,inputs=tf.concat([state_input,action_input],2),dtype=tf.float32)

            W3 = tf.Variable(tf.random_uniform([lstm_cell.output_size,1],-3e-3,3e-3))
            b3 = tf.Variable(tf.random_uniform([1],-3e-3,3e-3))
            q_value_output = tf.identity(tf.matmul(layer2,W3) + b3)
            net = [v for v in tf.trainable_variables() if scope in v.name]
        return state_input,action_input,q_value_output,net

    def create_target_q_network(self,state_dim,action_dim,net,scope):
       
        state_input,action_input,q_value_output,target_net = self.create_q_network(state_dim,action_dim,scope) 
        target_update = []
        for i in len(target_net):
            # theta' <-- tau*theta + (1-tau)*theta'
            target_update.append(target_net[i].assign(tf.add(tf.multiply(TAU,net[i]),tf.multiply((1-TAU),target[i]))))
        return state_input,action_input,q_value_output,target_update

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self,y_batch,state_batch,action_batch):
        self.time_step += 1
        self.sess.run(self.optimizer,feed_dict={
            self.y_input:y_batch,
            self.state_input:state_batch,
            self.action_input:action_batch
            })

    def gradients(self,state_batch,action_batch):
        return self.sess.run(self.action_gradients,feed_dict={
            self.state_input:state_batch,
            self.action_input:action_batch
            })[0]

    def target_q(self,state_batch,action_batch):
        return self.sess.run(self.target_q_value_output,feed_dict={
            self.target_state_input:state_batch,
            self.target_action_input:action_batch
            })

    def q_value(self,state_batch,action_batch):
        return self.sess.run(self.q_value_output,feed_dict={
            self.state_input:state_batch,
            self.action_input:action_batch})

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