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
            >>> q_network = rm.Sequential([rm.Dense(32),
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

        self.num_step = 0

        self.animation = Animation()
        self.test_mode = False

    def reset(self):
        return self.env.reset()

    def sample(self):
        return self.env.action_space.sample()

    def step(self, action):
        state, _, terminal, _ = self.env.step(int(action))
        self.num_step += 1

        reward = 0

        if terminal:
            reward = 1 if self.num_step >= 195 else -1
            self.num_step = 0

        return state, reward, terminal

    def test_start(self):
        self.animation.reset()
        self.env.reset()

    def test_epoch_step(self):
        self.animation.store(self.env.render(mode="rgb_array"))

    def test_close(self):
        # self.env.close()
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
            >>> q_network = rm.Sequential([rm.Dense(150),
            ... rm.Relu(),
            ... rm.Dense(100),
            ... rm.Relu(),
            ... rm.Dense(env.action_shape[0])
            ... ])
            >>> agent = DoubleDQN(env, q_network)
            >>> agent.fit()
    """

    def __init__(self):
        self.env = gym.make('Breakout-ramNoFrameskip-v4')
        self.action_shape = (self.env.action_space.n,)
        self.state_shape = self.env.observation_space.shape
        print("Env Space : ", self.state_shape)
        print("Action Space : ", self.action_shape)

        self.action_interval = 4
        self.real_done = True
        self.lives = 0
        self.max_time_length = 10000
        self.time_step = 0

        self.animation = Animation(ratio=36.0)

    def reset(self):
        if self.real_done:
            obs = self.env.reset()
            n_step = np.random.randint(5)
            for _ in range(n_step):
                obs, _, _ = self.step(0)
        else:
            obs, _, _ = self.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

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
        self.real_done = False
        self.time_step += 1

        for _ in range(self.action_interval):
            state, reward, terminal, _ = self.env.step(int(action))
            state_list.append(state)
            reward_list.append(reward)

            if terminal:
                self.real_done = True
                self.time_step = 0
                break

        processed_state = self._concat_state(state_list)

        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            terminal = True
            self.time_step = 0
        self.lives = lives

        if self.time_step > self.max_time_length:
            self.real_done = True
            terminal = True
            self.time_step = 0

        return processed_state, sum(reward_list), terminal

    def test_start(self):
        self.animation.reset()
        self.env.reset()

    def test_epoch_step(self):
        self.animation.store(self.env.render(mode="rgb_array"))

    def test_close(self):
        # self.env.close()
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
            >>> q_network = rm.Sequential([rm.Conv2d(16, filter=8, stride=4),
            ... rm.Relu(),
            ... rm.Conv2d(32, filter=4, stride=2),
            ... rm.Relu(),
            ... rm.Flatten(),
            ... rm.Dense(256),
            ... rm.Relu(),
            ... rm.Dense(env.action_shape[0])
            ... ])
            >>> agent = DoubleDQN(env, q_network)
            >>> agent.fit()
    """

    def __init__(self):
        self.env = gym.make('BreakoutNoFrameskip-v4')
        self.action_shape = (self.env.action_space.n,)
        self.action_interval = 4
        self.state_shape = (self.action_interval, 84, 84)
        print("Env space : ", self.env.observation_space.shape)
        print("Preprocessed Env space : ", self.state_shape)
        print("Action space : ", self.action_shape)

        self.real_done = True
        self.lives = 0
        self.time_step = 0
        self.max_time_length = 5000

        self.animation = Animation(ratio=36.0)

    def reset(self):
        if self.real_done:
            obs = self.env.reset()
            n_step = np.random.randint(5)
            for _ in range(n_step):
                obs, _, _ = self.step(0)
        else:
            obs, _, _ = self.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

    def sample(self):
        return self.env.action_space.sample()

    def _preprocess(self, state_list):
        processed_state_list = []
        for s in state_list:
            processed_state = np.uint8(resize(rgb2gray(s) * 255, (84, 84)))
            processed_state_list.append(processed_state)

        if len(processed_state_list) is not self.action_interval:
            for _ in range(self.action_interval - len(processed_state_list)):
                processed_state_list.append(processed_state_list[-1])

        return np.array(processed_state_list)

    def step(self, action):
        state_list = []
        reward_list = []
        self.real_done = False
        self.time_step += 1

        for _ in range(self.action_interval):
            state, reward, terminal, _ = self.env.step(int(action))
            state_list.append(state)
            reward_list.append(reward)

            if terminal:
                self.real_done = True
                self.time_step = 0
                break

        processed_state = self._preprocess(state_list)

        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            terminal = True
            self.time_step = 0
        self.lives = lives

        if self.time_step > self.max_time_length:
            self.real_done = True
            terminal = True
            self.time_step = 0

        return processed_state, sum(reward_list), terminal

    def test_start(self):
        self.animation.reset()
        self.env.reset()

    def test_epoch_step(self):
        self.animation.store(self.env.render(mode="rgb_array"))

    def test_close(self):
        # self.env.close()
        self.env.viewer = None
