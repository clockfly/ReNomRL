#!/usr/bin/env python
# coding: utf-8

import gym
import numpy as np
import matplotlib.pyplot as plt

import renom as rm
from renom_rl.discrete.double_dqn import DoubleDQN
from renom_rl.environ.env import BaseEnv
from renom_rl.utility import Animation


class CartPole(BaseEnv):
    """A wrapper environment of OpenAI gym "CartPole-v0"
    
        Description : https://github.com/openai/gym/wiki/CartPole-v0
        
        Example:
            >>> import renom as rm
            >>> from renom_rl.discrete.double_dqn import DoubleDQN
            >>> from renom_rl.environ.openai_env import CartPole
            >>> env = CartPole()
            Env Space :  (4,)
            Action Space :  (2,)
            >>> q_network = rm.Sequential([rm.Dense(10),
            ... rm.Relu(),
            ... rm.Dense(10),
            ... rm.Relu(),
            ... rm.Dense(env.action_shape[0])
            ... ])
            >>> agent = DoubleDQN(env, q_network)
            >>> agent.fit()
    """
    
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        
        self.action_shape = (self.env.action_space.n, )
        self.state_shape = self.env.observation_space.shape
        print("Env Space : ", self.state_shape)
        print("Action Space : ", self.action_shape)

        self.animation = Animation()
        self.test_mode = False
    
    def reset(self):
        return self.env.reset()
    
    def sample(self):
        return self.env.action_space.sample()
    
    def step(self, action):
        state, reward, terminal, _ = self.env.step(int(action))
        
        if self.test_mode==True:
            image = self.env.render(mode='rgb_array')
            self.animation.store(image)
        
        return state, reward, terminal

    def test_start(self):
        self.animation.reset()
        self.env.reset()
    
    def test_epoch_step(self):
        self.animation.store(self.env.render(mode="rgb_array"))
    
    def test_close(self):
        #self.env.close() 
        self.env.viewer = None

class MountainCar(BaseEnv):
	"""A wrapper environment of OpenAI gym "MountainCar-v0"
    
        Description : https://github.com/openai/gym/wiki/MountainCar-v0
        
        Example:
            >>> import renom as rm
            >>> from renom_rl.discrete.double_dqn import DoubleDQN
            >>> from renom_rl.environ.openai_env import MountainCar
            >>> env = MountainCar()
            Env Space :  (2,)
            Action Space :  (3,)
            >>> q_network = rm.Sequential([rm.Dense(30),
			... rm.Relu(),
			... rm.Dense(30),
			... rm.Relu(),
			... rm.Dense(env.action_shape[0])
			... ])
            >>> agent = DoubleDQN(env, q_network)
            >>> agent.fit()
    """
	
	def __init__(self):
		self.env = gym.make('MountainCar-v0') 
		self.action_shape = (self.env.action_space.n,)
		self.state_shape = self.env.observation_space.shape
		print("Env Space : ", self.state_shape)
		print("Action Space : ", self.action_shape)

		self.animation = Animation()
		self.test_mode = False
	
	def reset(self):
		return self.env.reset()
		
	def sample(self):
		return self.env.action_space.sample()

	def step(self, action):
		state, _, terminal, _ = self.env.step(int(action))
		reward = 0

		if terminal:
			if state[0] > 0.5:
				reward = 1
			else:
				reward = state[0] + 0.5

		return state, reward, terminal

	def test_start(self):
		self.animation.reset()
		self.env.reset()

	def test_epoch_step(self):
		self.animation.store(self.env.render(mode="rgb_array"))
		
	def test_close(self):
		#self.env.close() 
		self.env.viewer = None


class Breakout_ram(BaseEnv):
    """A wrapper environment of OpenAI gym "Breakout-ram-v0"
    
        Description : https://gym.openai.com/envs/Breakout-ram-v0/
        
        Example:
            >>> import renom as rm
            >>> from renom_rl.discrete.double_dqn import DoubleDQN
            >>> from renom_rl.environ.openai_env import Breakout_ram
            >>> env = Breakout_ram()
            Env Space :  (128,)
            Action Space :  (4,)
            >>> q_network = rm.Sequential([rm.Dense(200),
            ... rm.Relu(),
            ... rm.Dense(200),
            ... rm.Relu(),
            ... rm.Dense(100),
            ... rm.Relu(),
            ... rm.Dense(50),
            ... rm.Relu(),
            ... rm.Dense(env.action_shape[0])
            ... ])
            >>> agent = DoubleDQN(env, q_network)
            >>> agent.fit()
    """

    def __init__(self):
        self.env = gym.make('Breakout-ram-v0')
        self.action_shape = (self.env.action_space.n,)
        self.state_shape = self.env.observation_space.shape
        print("Env Space : ", self.state_shape)
        print("Action Space : ", self.action_shape)
        
        self.action_interval = 4

        self.animation = Animation(ratio=36.0)
        self.test_mode = False
    
    def reset(self):
        state = self.env.reset()
        n_step = np.random.randint(30)
        for _ in range(n_step):
            state, _, _ = self.step(self.env.action_space.sample())
        return state
    
    def sample(self):
        return self.env.action_space.sample()
    
    def _concat_state(self, state_list):
        max_ram_data = np.zeros(state_list[0].shape)
        for d in state_list:
            max_ram_data = np.maximum(d, max_ram_data)
        max_ram_data /= 255.0
        return max_ram_data
    
    def step(self, action):
        state_list = []
        reward_list = []
        
        for _ in range(self.action_interval):
            state, reward, terminal, _ = self.env.step(int(action))
            state_list.append(state)
            reward_list.append(reward)
            
            if terminal:
                break
        
        processed_state = self._concat_state(state_list)
        
        return processed_state, sum(reward_list), terminal

    def test_start(self):
        self.animation.reset()
        self.env.reset()
    
    def test_epoch_step(self):
        self.animation.store(self.env.render(mode="rgb_array"))
    
    def test_close(self):
        #self.env.close() 
        self.env.viewer = None 
        

class Breakout(BaseEnv):
    """A wrapper environment of OpenAI gym "Breakout-v0"
    
        Description : https://gym.openai.com/envs/Breakout-v0/
        
        Example:
            >>> import renom as rm
            >>> from renom_rl.discrete.double_dqn import DoubleDQN
            >>> from renom_rl.environ.openai_env import Breakout
            >>> env = Breakout()
            Env space :  (210, 160, 3)
            Preprocessed Env space :  (1, 84, 84)
            Action space :  (4,)
            >>> q_network = rm.Sequential([rm.Conv2d(32, filter=8, stride=4),
            ... rm.Relu(),
            ... rm.Conv2d(64, filter=4, stride=2),
            ... rm.Relu(),
            ... rm.Conv2d(64, filter=3, stride=1),
            ... rm.Relu(),
            ... rm.Flatten(),
            ... rm.Dense(512),
            ... rm.Relu(),
            ... rm.Dense(env.action_shape[0])
            ... ])
            >>> agent = DoubleDQN(env, q_network)
            >>> agent.fit()
    """
    
    def __init__(self):
        self.env = gym.make('Breakout-v0')
        self.action_shape = (self.env.action_space.n,)
        self.state_shape = (1, 84, 84)
        print("Env space : ", self.env.observation_space.shape)
        print("Preprocessed Env space : ", self.state_shape)
        print("Action space : ", self.action_shape)
        
        self.action_interval = 4
        
        self.animation = Animation(ratio=36.0)
        self.test_mode = False
        
    def reset(self):
        state = self.env.reset()
        n_step = np.random.randint(30)
        for _ in range(n_step):
            state, _, _ = self.step(self.env.action_space.sample())
        return state
    
    def sample(self):
        return self.env.action_space.sample()
    
   
    def _preprocess(self, state_list):
        max_pixels = np.zeros(state_list[0].shape)
        for s in state_list:
            max_pixels = np.maximum(s, max_pixels)
        processed_state = resize(rgb2gray(max_pixels), (1, 84, 84))
        return processed_state
        
    def step(self, action):
        state_list = []
        reward_list = []
        
        for _ in range(self.action_interval):
            state, reward, terminal, _ = self.env.step(int(action))
            state_list.append(state)
            reward_list.append(reward)
            
            if terminal:
                break
                
        processed_state = self._preprocess(state_list)
        
        return processed_state, sum(reward_list), terminal
    
    def test_start(self):
        self.animation.reset()
        self.env.reset()
    
    def test_epoch_step(self):
        self.animation.store(self.env.render(mode="rgb_array"))
    
    def test_close(self):
        #self.env.close() 
        self.env.viewer = None