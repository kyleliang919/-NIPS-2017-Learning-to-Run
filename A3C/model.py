from ou_noise import OUNoise

from helper import *

import opensim as osim
from osim.http.client import Client
from osim.env import *

import threading
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from helper import *

from time import sleep
from time import time
from time import gmtime, strftime
import multiprocessing
from multiprocessing import Process, Pipe
from osim.env import *

# [Hacked] the memory might always be leaking, here's a solution #58
# https://github.com/stanfordnmbl/osim-rl/issues/58 
# separate process that holds a separate RunEnv instance.
# This has to be done since RunEnv() in the same process result in interleaved running of simulations.
def floatify(np):
    return [float(np[i]) for i in range(len(np))]
    
def standalone_headless_isolated(conn,vis):
    e = RunEnv(visualize=vis)
    
    while True:
        try:
            msg = conn.recv()

            # messages should be tuples,
            # msg[0] should be string

            if msg[0] == 'reset':
                o = e.reset(difficulty=2)
                conn.send(o)
            elif msg[0] == 'step':
                ordi = e.step(msg[1])
                conn.send(ordi)
            else:
                conn.close()
                del e
                return
        except:
            conn.close()
            del e
            raise

# class that manages the interprocess communication and expose itself as a RunEnv.
class ei: # Environment Instance
    def __init__(self,vis):
        self.pc, self.cc = Pipe()
        self.p = Process(
            target = standalone_headless_isolated,
            args=(self.cc,vis,)
        )
        self.p.daemon = True
        self.p.start()

    def reset(self):
        self.pc.send(('reset',))
        return self.pc.recv()

    def step(self,actions):
        self.pc.send(('step',actions,))
        try:
	    return self.pc.recv()
	except EOFError:  
            return None

    def __del__(self):
        self.pc.send(('exit',))
        #print('(ei)waiting for join...')
        self.p.join()


# ================================================================
# Model components
# ================================================================

# Actor Network------------------------------------------------------------------------------------------------------------
class AC_Network():
    def __init__(self,s_size,a_size,scope,trainer):
        with tf.variable_scope(scope):
            #Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            
	        # Create the model, use the default arg scope to configure the batch norm parameters.
            '''
            conv1 = tf.nn.elu(tf.nn.conv1d(self.imageIn,tf.truncated_normal([2,1,8],stddev=0.1),2,padding='VALID'))
            conv2 = tf.nn.elu(tf.nn.conv1d(conv1,tf.truncated_normal([3,8,16],stddev=0.05),1,padding='VALID'))
        
            hidden = slim.fully_connected(slim.flatten(conv2),200,activation_fn=tf.nn.elu)'''
            
            hidden1 = slim.fully_connected(self.inputs,300,activation_fn=tf.elu)
            hidden2 = slim.fully_connected(hidden1,200,activation_fn=tf.elu)
    
            #Output layers for policy and value estimations
            mu = slim.fully_connected(hidden2,a_size,activation_fn=tf.sigmoid,weights_initializer=normalized_columns_initializer(0.01),biases_initializer=None)
            var = slim.fully_connected(hidden2,a_size,activation_fn=tf.nn.softplus,weights_initializer=normalized_columns_initializer(0.01),biases_initializer=None)
            self.normal_dist = tf.contrib.distributions.Normal(mu, 0.05)
            self.policy = tf.clip_by_value(self.normal_dist.sample(1),0.0,1.0) # self.normal_dist.sample(1)
            self.value = slim.fully_connected(hidden2,1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)
                
            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None,a_size],dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.log_prob = tf.reduce_sum(self.normal_dist.log_prob(self.actions),axis=1)
                self.entropy = tf.reduce_sum(self.normal_dist.entropy(),axis=1)  # encourage exploration
                self.entropy = tf.reduce_sum(self.entropy,axis=0)
                self.policy_loss = -tf.reduce_sum(self.log_prob*self.advantages,axis=0)

                self.loss = 0.5 * self.value_loss + self.policy_loss - 0.01 * self.entropy

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)

                #Apply local gradients to global network
                #Comment these two lines out to stop training
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))
                
# Learning to run Worker------------------------------------------------------------------------------------------------------------
class Worker():
    def __init__(self,name,s_size,a_size,trainer,model_path,global_episodes,is_training):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_"+str(self.number))
        self.is_training = is_training

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size,a_size,self.name,trainer)
	    #self.local_AC_target = AC_Network(s_size,a_size,self.name+'/target',trainer,noisy)
        self.update_local_ops = update_target_graph('global',self.name)
	    #self.update_local_ops_target = update_target_graph('global/target',self.name+'/target')
	    #self.update_global_target = update_target_network(self.name,'global/target')           
        
	    # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OUNoise(a_size)
        
    def train(self,rollout,sess,gamma,bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
    	# reward clipping:  scale and clip the values of the rewards to the range -1,+1
    	#rewards = (rewards - np.mean(rewards)) / np.max(abs(rewards))

        next_observations = rollout[:,3] # Aug 1st, notice next observation is never used
        values = rollout[:,5]
        
        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v:discounted_rewards,
            self.local_AC.inputs:np.vstack(observations),
            self.local_AC.actions:np.vstack(actions),
            self.local_AC.advantages:advantages}
        l,v_l,p_l,e_l,g_n,v_n,_ = sess.run([self.local_AC.loss,self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        return l / len(rollout), v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n,v_n
        
    def work(self,max_episode_length,gamma,sess,coord,saver):
        if self.is_training:
            episode_count = sess.run(self.global_episodes)
        else:
            episode_count = 0
        wining_episode_count = 0
        total_steps = 0
        print ("Starting worker " + str(self.number))
        with open('result.txt','w') as f:
            f.write(strftime("Starting time: %a, %d %b %Y %H:%M:%S\n", gmtime()))
	
        explore = 1000

        if self.name == 'worker_1':
            self.env = ei(vis=False)#RunEnv(visualize=True)
        else:
            self.env = ei(vis=False)#RunEnv(visualize=False)

        with sess.as_default(), sess.graph.as_default():
            #not_start_training_yet = True
            while not coord.should_stop():
                # start the env (in the thread) every 50 eps to prevent memory leak
                if episode_count % 50 == 0:
                    if self.env != None:
                        del self.env
                    if self.name == 'worker_1':
                        self.env = ei(vis=True)#RunEnv(visualize=True)
                    else:
                        self.env = ei(vis=False)#RunEnv(visualize=False)
                self.setting=2
                        
                sess.run(self.update_local_ops)
		        #sess.run(self.update_local_ops_target)
                episode_buffer = []
                episode_values = []
                episode_reward = 0
                episode_step_count = 0
                done = False
                
                seed = np.random.rand()
                
                self.env.reset()
                # engineered initial input to make agent's life easier
                a=engineered_action(seed)
                ob = self.env.step(a)[0]
                s = ob
                ob = self.env.step(a)[0]
                s1 = ob
                s = process_state(s,s1)
                explore -= 1
                #st = time()
                chese=0
                while done == False:
                    #Take an action using probabilities from policy network output.
                    action,v = sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out], 
                                feed_dict={self.local_AC.inputs:[s])
                    if not (episode_count % 5 == 0 and self.name == 'worker_1') and self.is_training:
                        if explore > 0: # > 0 turn on OU_noise # test the agent every 2 eps
	                        a = np.clip(action[0,0]+self.exploration_noise.noise(),0.0,1.0)
                        else:
                            a = action[0,0]
                        if chese < 60 and episode_count < 250:
                            a=engineered_action(seed)
                            chese += 1
                    else:
                        a = action[0,0]
                    ob,r,done,_ = self.env.step(a)
                    '''
                    if self.name == 'worker_0':
                        ct = time()
                        print(ct-st)
                        st = ct
                    '''
                    if done == False:
                        s2 = ob
                    else:
                        s2 = s1
                    s1 = process_state(s1,s2)
                    #print(s1)    
                    episode_buffer.append([s,a,r,s1,done,v[0,0]])
                    episode_values.append(v[0,0])

                    episode_reward += r
                    s = s1
                    s1 = s2
                    total_steps += 1
                    episode_step_count += 1
                            
                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    '''        
                    if len(episode_buffer) == 120 and done != True and episode_step_count != max_episode_length - 1: # change pisode length to 5, and try to modify Worker.train() function to utilize the next frame to train imagined frame.
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        if self.is_training:
                            v1 = sess.run(self.local_AC.value, 
                            feed_dict={self.local_AC.inputs:[s]})[0,0]
                            l,v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,v1)
                            sess.run(self.update_local_ops)
                            episode_buffer = []
                    ''' 
                    if done == True:
                        break
                           
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))
                    
                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    if self.is_training:
                        l,v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,0.0)
                        #print(l,v_l,p_l,e_l,g_n,v_n)
	                    #sess.run(self.update_global_target)
                                    
                        
                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    if self.is_training:
                        summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                        summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                        summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.flush()
                    if self.name == 'worker_1':
			with open('result.txt','a') as f:
                            f.write("Episode "+str(episode_count)+" reward (testing): %.2f\n" % episode_reward)
                    if self.name == 'worker_0':
			with open('result.txt','a') as f:
			    f.write("Episodes "+str(episode_count)+" mean reward (training): %.2f\n" % mean_reward)

                        if episode_count % 100 == 0:
                            saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                            with open('result.txt','a') as f:
			        f.write("Saved Model at episode: "+str(episode_count)+"\n")
                if self.name == 'worker_0' and self.is_training:
                    sess.run(self.increment)
                        
                episode_count += 1
                    
                if self.name == "worker_1" and episode_reward > 2.:
                    wining_episode_count += 1
                    print('Worker_1 is stepping forward in Episode {}! Reward: {:.2f}. Total percentage of success is: {}%'.format(episode_count, episode_reward, int(wining_episode_count / episode_count * 100)))
                    with open('result.txt','a') as f:
			f.wirte('Worker_1 is stepping forward in Episode {}! Reward: {:.2f}. Total percentage of success is: {}%\n'.format(episode_count, episode_reward, int(wining_episode_count / episode_count * 100)))
        
        # All done Stop trail
        # Confirm exit
        print('Exit/Done '+self.name)
