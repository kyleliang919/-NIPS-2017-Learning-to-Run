import numpy as np
import tensorflow as tf
from utils import *
from model import *
import argparse
from rollouts import *
import json
import opensim as osim
from osim.http.client import Client
from osim.env import *
from wrapperEnv import WrapperEnv
parser = argparse.ArgumentParser(description='TRPO.')
# these parameters should stay the same
parser.add_argument("--task", type=str, default='Reacher-v1')
parser.add_argument("--timesteps_per_batch", type=int, default=10000)
parser.add_argument("--n_steps", type=int, default=6000000)
parser.add_argument("--gamma", type=float, default=.99)
parser.add_argument("--max_kl", type=float, default=.001)
parser.add_argument("--cg_damping", type=float, default=1e-3)
parser.add_argument("--num_threads", type=int, default=5)
parser.add_argument("--monitor", type=bool, default=False)

# change these parameters for testing
parser.add_argument("--decay_method", type=str, default="adaptive") # adaptive, none
parser.add_argument("--timestep_adapt", type=int, default=0)
parser.add_argument("--kl_adapt", type=float, default=0)

args = parser.parse_args()
args.max_pathlength = 1000


learner_tasks = multiprocessing.JoinableQueue()
learner_results = multiprocessing.Queue()
learner_env = WrapperEnv(visualize=False)

learner = TRPO(args, learner_env.observation_space_shape, learner_env.action_space, learner_tasks, learner_results)
learner.makeModel()
learner.loadModel(0)
# print learner.act([[1]*41])


def main():
	env = learner_env
	agent = learner
	observation = env.reset(difficulty=2)

	# Run a single step
	#
	# The grader runs 3 simulations of at most 1000 steps each. We stop after the last one
	total_reward = 0
	while True:
		action = agent.act([observation]).tolist()
		[observation, reward, done, info] = env.step(action)
		total_reward = total_reward + reward
		print observation
		print "reward of this step:"+str(reward)
		if done:
			break
	print "total reward:"+str(total_reward)

if __name__ == '__main__':
    main()
