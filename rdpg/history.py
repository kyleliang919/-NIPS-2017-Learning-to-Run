class History(object):
	def __init__(self,init_state):
		self.observations = [init_state]
		self.actions = []
		self.rewards = []
	# append observation
	def append_o(self,state):
		self.observations.append(state)
	# append action
	def append_a(self,action):
		self.actions.append(action)
	# append reward
	def append_r(self,reward):
		self.rewards.append(reward)
	def append(self,state,action,reward):
		self.append_o(state)
		self.append_a(action)
		self.append_r(reward)
