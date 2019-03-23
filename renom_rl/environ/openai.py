#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A wrapper environment of OpenAI gym.

| -**OpenAI Gym**
| https://github.com/openai/gym
|

Example:
    >>> import renom as rm
    >>> from renom_rl.discrete.dqn import DQN
    >>> from renom_rl.environ.openai import Pendulum
    >>> env = Pendulum()
    >>> model = rm.Sequential([
    ...     rm.Dense(10),
    ...     rm.Dense(1),
    ... ])
    ...
    >>> agent = DQN(env, model)
    >>> agent.fit()
"""


import gym
import numpy as np
from PIL import Image
from .env import BaseEnv
import cv2


class Pendulum(BaseEnv):
    """
    OpenAI gym "Pendulum".

    """

    def __init__(self):
        self.env = gym.make('Pendulum-v0')
        self.action_shape = (1,)
        self.state_shape = (3,)
        super(Pendulum, self).__init__()

    def reset(self):
        """"""
        return self.env.reset().reshape(3)

    def sample(self):
        """"""
        return self.env.action_space.sample()

    def step(self, action):
        """"""
        state, reward, terminal = self.env.step(action)[:3]
        return state.reshape(3), reward, terminal


cv2.ocl.setUseOpenCL(False)


class Breakout(BaseEnv):
    """A wrapper environment of OpenAI gym "BreakoutNoFrameskip-v4"

        Description : https://gym.openai.com/envs/Breakout-v0/

        Example:
            >>> import renom as rm
            >>> from renom_rl.discrete.double_dqn import DoubleDQN
            >>> from renom_rl.environ.openai_env import Breakout
            >>> env = Breakout()
            Env space :  (210, 160, 3)
            Preprocessed Env space :  (4, 84, 84)
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
        self.action_shape = (4,)
        self.action_interval = 4
        self.state_shape = (4, 84, 84)
        self.test = False
        self.lives = 5
        self.true_terminal = True
        self.test_mode = False
        self.previous_frames = np.zeros(self.state_shape)

        print("Env space :  ", (210, 160, 3))
        print("Preprocessed Env space :  ", self.state_shape)
        print("Action space :  ", self.action_shape)

    def _preprocess(self, state):
        frame = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, self.state_shape[1:],
                           interpolation=cv2.INTER_AREA)

        return frame

    def get_step(self, action):
        # getting image of trajectory as list
        state_list = []
        reward_list = []
        weight_interval = 0.2

        for x in range(self.action_interval):
            state, reward, terminal, info = self.env.step(int(action))
            state_list.append(state)
            reward_list.append(reward)

        # getting max of each layer, respect to each pixel
        state = np.stack(state_list[-2:]).max(axis=0)
        reward = sum(reward_list)

        state = self._preprocess(state)

        return state, reward, terminal, info

    def append_and_get(self, state):

        self.previous_frames[:-1] = self.previous_frames[1:].copy()
        self.previous_frames[-1] = state.copy() / 255.0

    def reset(self):

        self.previous_frames = np.zeros((4, 84, 84))

        if self.true_terminal:
            self.env.reset()
            self.lives = 5
            self.true_terminal = False

        action_list = np.random.randint(0, 4, size=10)
        fire = np.array([1])
        action_rand = np.zeros((np.random.randint(30)))

        action_list_all = np.concatenate([action_list, fire, action_rand])

        for a in action_list_all:
            _ = self.env.step(int(a))

        state, _, _, _ = self.get_step(0)
        self.append_and_get(state)

        state_final = self.previous_frames

        return state_final

    def sample(self):
        return self.env.action_space.sample()

    def step(self, action):

        state, reward, _, info = self.get_step(int(action))

        lives = info["ale.lives"]
        terminal = False
        self.append_and_get(state)

        # getting lives
        if lives > 0:
            if lives < self.lives:
                self.lives = lives
                terminal = True
        else:
            terminal = True
            self.true_terminal = True

        state_final = np.stack(self.previous_frames)

        return state_final, reward, terminal

    def epoch(self):
        self.true_terminal = True


class CartPole00(BaseEnv):

    """A wrapper environment of OpenAI gym "CartPole-v0".
    When 195 step is stepped without error, then reward is +1, else -1.
    """

    def __init__(self):
        self.action_shape = (2,)
        self.state_shape = (4,)

        self.env = gym.make('CartPole-v0')
        self.step_continue = 0
        self.test_mode = False
        self.reward = 0

    def reset(self):
        return self.env.reset()

    def sample(self):
        rand = self.env.action_space.sample()
        return rand

    def step(self, action):
        state, _, terminal, _ = self.env.step(int(action))

        self.step_continue += 1
        reward = 0

        if terminal:
            if self.step_continue >= 195:
                reward = 1
            else:
                reward = -1

            self.step_continue = 0

        self.reward = reward

        return state, reward, terminal


class CartPole01(BaseEnv):

    """A wrapper environment of OpenAI gym "CartPole-v0".
    When 195 step is stepped without error, then reward is +1, else -1.
    Training loop brakes when reward +1 is recieved 10 times.
    """

    def __init__(self):
        self.action_shape = (2,)
        self.state_shape = (4,)

        self.env = gym.make('CartPole-v0')
        self.step_continue = 0
        self.successful_episode = 0
        self.test_mode = False
        self.reward = 0

    def reset(self):
        return self.env.reset()

    def sample(self):
        rand = self.env.action_space.sample()
        return rand

    def step(self, action):
        state, _, terminal, _ = self.env.step(int(action))

        self.step_continue += 1
        reward = 0

        if terminal:
            if self.step_continue >= 195:
                reward = 1
                if self.test_mode == False:
                    print(self.successful_episode)
                    self.successful_episode += 1
            else:
                reward = -1
            self.step_continue = 0

        self.reward = reward

        return state, reward, terminal

    def terminate(self):
        if self.successful_episode >= 10:
            self.successful_episode = 0
            return True
        else:
            return False

    def test_start(self):
        self.test_mode = True

    def test_close(self):
        self.env.close()
        self.env.viewer = None
        self.test_mode = False


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
        # self.env.close()
        self.env.viewer = None
