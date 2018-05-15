from __future__ import print_function
import numpy as np


CONTINUOUS = 0
DISCRETE = 1
class OU(object):
    """ 
    DDPG paper ornstein-uhlenbeck noise parameters are theta=0.15, sigma=0.2 
    """

    def __init__(self, mu=0, theta=0.15, sigma=0.2):
        self.noise_type = CONTINUOUS
        self.mu = mu
        self.theta = theta
        self.sigma = sigma

    def sample(self, action):
        shape = getattr(self.mu, 'shape', [1,])
        return self.theta * (self.mu - action) + self.sigma \
            * np.random.randn(*shape)

class GP(object):

    def __init__(self, mean=0, std=0.1):
        self.noise_type = CONTINUOUS
        self._mean = mean
        self._std = std

    def samplse(self, action):
        shape = getattr(action, 'shape', [1,])
        return self._mean + self._std*np.random.randn(*shape)
