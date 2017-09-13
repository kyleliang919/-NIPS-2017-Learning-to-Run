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
import time
class WrapperEnv():
	def __init__(self,visualize=False):
		self.env = RunEnv(visualize)
		self.ob_0 = self.preprocess(np.array(self.env.reset(difficulty=2)))
		self.ob_1 = np.zeros(14)
		# self.ob_2 = np.zeros(41)
		self.observation_space_shape = (55,)
		self.action_space = self.env.action_space
		self.difficulty = 0
	def reset(self,difficulty=2):
		self.ob_0 = self.preprocess(np.array(self.env.reset(difficulty)))
		self.difficulty = difficulty
		if self.difficulty == 0:
			self.ob_0[38] = 0
		self.ob_0[1] = 0
		self.ob_1 = np.zeros(14)
		# self.ob_2 = np.zeros(41)
		# return np.concatenate((self.ob_0,self.ob_1,self.ob_2),axis=0)
		return np.concatenate((self.ob_0,self.ob_1),axis=0)

	def step(self,action):
		res=self.env.step(action)
		ob_0_post = self.ob_0
		# ob_1_post = self.ob_1
		# ob_2_post = self.ob_2
		self.ob_0 = self.preprocess(np.array(res[0]))
		self.ob_0[1]=0
		self.ob_1 = (self.ob_0[22:36] - ob_0_post[22:36])/0.01
		if self.difficulty == 0:
			self.ob_0[38]=0
		# self.ob_2 = self.ob_1 - ob_1_post
		# res[0] = np.concatenate((self.ob_0,self.ob_1,self.ob_2),axis=0)
		res[0] = np.concatenate((self.ob_0,self.ob_1),axis=0)
		return res

	def seed(self,s):
		self.env.seed(s)

	def preprocess(self,v):
		n = [1,18,22,24,26,28,30,32,34]
		m = [19,23,25,27,29,31,33,35]
		for i in n:
			v[i]=v[i]-v[1]
		for i in m:
			v[i]=v[i]-v[2]
		v[20] = v[20]-v[4]
		v[21] = v[21]-v[5]
		return v
