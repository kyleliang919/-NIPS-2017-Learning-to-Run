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

class WrapperClient():
	def __init__(self,remote_base):
		self.client = Client(remote_base)
		self.ob_0 = np.array(41)
		self.ob_1 = np.zeros(14)
		# self.ob_2 = np.zeros(41)

	def env_create(self,token):
		self.ob_0 = self.preprocess(np.array(self.client.env_create("7be35dd3a64deac826068d37c2258847")))
		# return np.concatenate((self.ob_0,self.ob_1,self.ob_2),axis=0)
		return np.concatenate((self.ob_0,self.ob_1),axis=0)

	def env_reset(self):
		ob = self.client.env_reset()
		if ob is None:
			return None
		self.ob_0 = self.preprocess(np.array(ob))
		self.ob_0[1] = 0
		self.ob_1 = np.zeros(14)
		# self.ob_2 = np.zeros(41)
		# return np.concatenate((self.ob_0,self.ob_1,self.ob_2),axis=0)
		return np.concatenate((self.ob_0,self.ob_1),axis=0)
		

	def env_step(self,action):
		res=self.client.env_step(action)
		ob_0_post = self.ob_0
		# ob_1_post = self.ob_1
		# ob_2_post = self.ob_2
		self.ob_0 = self.preprocess(np.array(res[0]))
		self.ob_0[1] = 0
		self.ob_1 = (self.ob_0[22:36] - ob_0_post[22:36])/0.01
		# self.ob_2 = self.ob_1 - ob_1_post
		# res[0] = np.concatenate((self.ob_0,self.ob_1,self.ob_2),axis=0)
		return np.concatenate((self.ob_0,self.ob_1),axis=0)
		return res

	def submit(self):
		self.client.submit()

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