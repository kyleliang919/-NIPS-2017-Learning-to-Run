# -----------------------------------
# Deep Deterministic Policy Gradient
# Author: Kaizhao Liang
# Date: 08.11.2017
# -----------------------------------
import tensorflow as tf
import numpy as np
from ou_noise import OUNoise
from critic_network import CriticNetwork
from actor_network import ActorNetwork
from replay_buffer import ReplayBuffer
from helper import *

import opensim as osim
from osim.http.client import Client
from osim.env import *

# Hyper Parameters:

REPLAY_BUFFER_SIZE = 1000000
REPLAY_START_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99


class Worker:
    """docstring for DDPG"""
    def __init__(self,sess,number,model_path,global_episodes,explore,decay,training):
        self.name = 'worker_' + str(number) # name for uploading results
	self.number = number
        # Randomly initialize actor network and critic network
        # with both their target networks
        self.state_dim = 41
        self.action_dim = 18
	self.model_path= model_path
	self.global_episodes = global_episodes
	self.increment = self.global_episodes.assign_add(1)
	self.sess = sess
	self.explore = explore
	self.decay = decay
	self.training = training


        self.actor_network = ActorNetwork(self.sess,self.state_dim,self.action_dim,self.name+'/actor')
	self.actor_network.update_target(self.sess)
        self.critic_network = CriticNetwork(self.sess,self.state_dim,self.action_dim,self.name+'/critic')
	self.critic_network.update_target(self.sess)

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OUNoise(self.action_dim)

	self.update_local_ops_actor = update_target_graph('global/actor',self.name+'/actor')
	self.update_local_ops_critic = update_target_graph('global/critic',self.name+'/critic')

    def start(self,setting=0):
	self.env = RunEnv(visualize=True)
	self.setting=setting

    def train(self):
        #print "train step",self.time_step
        # Sample a random minibatch of N transitions from replay buffer
        minibatch = self.replay_buffer.get_batch(BATCH_SIZE)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])

        # for action_dim = 1
        action_batch = np.resize(action_batch,[BATCH_SIZE,self.action_dim])

        # Calculate y_batch

        next_action_batch = self.actor_network.target_actions(self.sess,next_state_batch)
        q_value_batch = self.critic_network.target_q(self.sess,next_state_batch,next_action_batch)
        y_batch = []
        for i in range(len(minibatch)):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else :
                y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])
        y_batch = np.resize(y_batch,[BATCH_SIZE,1])
        # Update critic by minimizing the loss L
        self.critic_network.train(y_batch,state_batch,action_batch)

        # Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_network.actions(self.sess,selfstate_batch)
        q_gradient_batch = self.critic_network.gradients(self.sess,state_batch,action_batch_for_gradients)

        self.actor_network.train(self.sess,q_gradient_batch,state_batch)

        # Update the target networks
        self.actor_network.update_target(self.sess)
        self.critic_network.update_target(self.sess)

    def save_model(self, saver, episode):
        #if self.episode % 10 == 1:
	if self.name == 'worker_0':
            saver.save(self.sess, self.model_path + "/model-" + str(episode) + ".ckpt")

    def noise_action(self,state,decay):
        # Select action a_t according to the current policy and exploration noise which gradually vanishes
        action = self.actor_network.action(self.sess,state)
        return action+self.exploration_noise.noise()*decay

    def action(self,state):
        action = self.actor_network.action(self.sess,state)
        return action

    def perceive(self,state,action,reward,next_state,done):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.replay_buffer.add(state,action,reward,next_state,done)

        # Store transitions to replay start size then start training
        if self.replay_buffer.count() >  REPLAY_START_SIZE and self.training:
            self.train()

        #if self.time_step % 10000 == 0:
            #self.actor_network.save_network(self.time_step)
            #self.critic_network.save_network(self.time_step)

        # Re-iniitialize the random process when an episode ends
        if done:
            self.exploration_noise.reset()

    def work(self,coord,saver):
        if self.training:
            episode_count = self.sess.run(self.global_episodes)
        else:
            episode_count = 0
        wining_episode_count = 0
        total_steps = 0
        print ("Starting worker_" + str(self.number))

        with self.sess.as_default(), self.sess.graph.as_default():
            #not_start_training_yet = True
            while not coord.should_stop():
		returns = []
    		rewards = []
		episode_reward = 0

		if np.random.rand() < 0.9: # change Aug20 stochastic apply noise
		    noisy = True
		    self.decay -= 1./self.explore
		else:
		    noisy = False

                
                self.sess.run(self.update_local_ops_actor)
                self.sess.run(self.update_local_ops_critic)
                
		state = self.env.reset(difficulty = self.setting)
		#print(observation)
		s = process_frame(state)
		    
		print "episode:",episode_count
		# Train

		for step in xrange(self.env.spec.timestep_limit):
		    state = process_frame(state)
		    if noisy:
		        action = np.clip(self.noise_action(state,np.maximum(self.decay,0)),0.0,1.0) # change Aug20, decay noise (no noise after ep>=self.explore)
		    else:
			action = self.action(state)
	            next_state,reward,done,_ = self.env.step(action)
	            #print('state={}, action={}, reward={}, next_state={}, done={}'.format(state, action, reward, next_state, done))
	            next_state = process_frame(next_state)
	            self.perceive(state,action,reward*100,next_state,done)
	            state = next_state
	            episode_reward += reward
	            if done:
	        	break

		if episode % 5 == 0:
	    	    print "episode reward:",reward_episode
		

		# Testing:
	        #if episode % 1 == 0:
	        if self.name == 'worker_0' and episode_count % 50 == 0 and episode_count > 1: # change Aug19
	            self.save_model(saver, episode_count)
	       	    total_return = 0
		    ave_reward = 0
	            for i in xrange(TEST):
	                state = self.env.reset()
	       	        reward_per_step = 0
		        for j in xrange(self.env.spec.timestep_limit):
		            action = self.action(process_frame(state)) # direct action for test
		            state,reward,done,_ = self.env.step(action)
		            total_return += reward
		        if done:
   	                    break
			    reward_per_step += (reward - reward_per_step)/(j+1)
			ave_reward += reward_per_step

		    ave_return = total_return/TEST
	            ave_reward = ave_reward/TEST
		    returns.append(ave_return)
	            rewards.append(ave_reward)

		    print 'episode: ',episode,'Evaluation Average Return:',ave_return, '  Evaluation Average Reward: ', ave_reward

		if self.name == 'worker_0' and self.training:
                    sess.run(self.increment)
		episode_count += 1

	    # All done Stop trail
            # Confirm exit
            print('Done '+self.name)








