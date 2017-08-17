# -----------------------------------
# Recurrent Deep Deterministic Policy Gradient
# Author: Kaizhao Liang
# Date: 08.11.2017
# -----------------------------------
import tensorflow as tf
import numpy as np
from ou_noise import OUNoise
from critic_network import CriticNetwork
from actor_network import ActorNetwork
from replay_buffer import ReplayBuffer
from history import History
# Hyper Parameters:

REPLAY_BUFFER_SIZE = 1000000
REPLAY_START_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.8


class RDPG:
    """docstring for RDPG"""
    def __init__(self, env):
        self.name = 'RDPG' # name for uploading results
        self.environment = env
        # Randomly initialize actor network and critic network
        # with both their target networks
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.sess = tf.InteractiveSession()

        self.actor_network = ActorNetwork(self.sess,self.state_dim,self.action_dim)
        self.critic_network = CriticNetwork(self.sess,self.state_dim,self.action_dim)

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OUNoise(self.action_dim)

        self.saver = tf.train.Saver()

    def train(self):
        # Sample a random minibatch of N sequences from replay buffer
        minibatch = self.replay_buffer.get_batch(BATCH_SIZE)
        # Construct histories
        observations = []
        next_observations = []
        actions = []
        rewards = []
        dones = []
        for each in minibatch:
            for i in range(1,len(each.observations)):
                observations.append(self.pad(each.observations[0:i]))
                next_observations.append(self.pad(each.observations[1,i+1]))
                actions.append(each.actions[0:i-1])
                rewards.append(each.rewards[0:i])
                if i == len(each.observations) - 1:
                    dones.append(True)
                else:
                    dones.append(False)
        # Calculate y_batch
        next_action_batch = self.actor_network.target_action(observations)
        q_value_batch = self.critic_network.target_q(next_observations,[self.pad(i+j) for (i,j) in zip(actions,next_action_batch)])
        y_batch = []
        for i in range(len(observations)):
            if dones[i]:
                y_batch.append(rewards[i][-1])
            else:
                y_batch.append(rewards[i][-1] + GAMMA * q_value_batch[i])
        y_batch = np.resize(y_batch,[len(observations),1])
        # Update critic by minimizing the loss L
        self.critic_network.train(y_batch,observations,[self.pad(i) for i in actions])

        # Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_network.actions(observations)
        q_gradient_batch = self.critic_network.gradients(observations,action_batch_for_gradients)

        self.actor_network.train(q_gradient_batch,observations)

        # Update the target networks
        self.actor_network.update_target()
        self.critic_network.update_target()

    def save_model(self, path, episode):
        self.saver.save(self.sess, path + "modle.ckpt", episode)


    def noise_action(self,history):
        # Select action a_t according to a sequence of observation and action
        action = self.actor_network.action(history)
        return action+self.exploration_noise.noise()

    def action(self,history):
        action = self.actor_network.action(history)
        return action

    def perceive(self,history):
        # Store the history sequence in the replay buffer
        self.replay_buffer.add(history)

        # Store history to replay start size then start training
        if self.replay_buffer.count() >  REPLAY_START_SIZE:
            self.train()

        # Re-iniitialize the random process when an episode ends
        if done:
            self.exploration_noise.reset()

    def pad(self,input):
        dim = len(input[0])
        return input+[[0]*dim]*(1000-len(input))










