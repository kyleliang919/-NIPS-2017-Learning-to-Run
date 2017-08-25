# --------------------------------------
# Ornstein-Uhlenbeck Noise
# Author: Flood Sung
# Date: 2016.5.4
# Reference: https://github.com/rllab/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py
# --------------------------------------

import numpy as np
import numpy.random as nr

class OUNoise:
    """docstring for OUNoise"""
    def __init__(self,action_dimension,mu=0.0, theta=0.02, sigma=0.07):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state

if __name__ == '__main__':
    ou = OUNoise(18)
    states = []
    for i in range(115):
        states.append([max(0.5,i) for i in ou.noise()])
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()
