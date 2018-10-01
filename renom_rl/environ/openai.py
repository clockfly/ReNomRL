from .env import BaseEnv
import numpy as np
from PIL import Image
import gym


class Pendulum(BaseEnv):

    def __init__(self):
        self.env = gym.make('Pendulum-v0')
        self.action_shape = 1
        self.state_shape = 3
        super(Pendulum, self).__init__()

    def reset(self):
        return self.env.reset().reshape(3)

    def sample(self):
        return self.env.action_space.sample()

    def step(self, action):
        state, reward, terminal = self.env.step(action)[:3]
        return state.reshape(3), reward, terminal

    def render(self):
        self.env.render()


class Breakout(BaseEnv):

    def __init__(self):
        self.env = gym.make('BreakoutNoFrameskip-v4')
        self.action_shape = (4,)
        self.state_shape = (4, 84, 84)
        self.previous_frames = []
        self._reset_flag = True
        self._last_live = 5
        super(Breakout, self).__init__()

    def reset(self):
        if self._reset_flag:
            self._reset_flag = False
            self.env.reset()
        n_step = np.random.randint(4, 32+1)
        for _ in range(n_step):
            state, _, _ = self.step(self.env.action_space.sample())
        return state

    def sample(self):
        return self.env.action_space.sample()

    def render(self):
        self.env.render()

    def _preprocess(self, state):
        resized_image = Image.fromarray(state).resize((84, 110)).convert('L')
        image_array = np.asarray(resized_image)/255.
        final_image = image_array[26:110]
        return final_image

    def step(self, action):
        state_list = []
        reward_list = []
        terminal = False
        for _ in range(4):
            # Use last frame. Other frames will be skipped.
            s, r, t, info = self.env.step(action)
            state = self._preprocess(s)
            reward_list.append(r)
            if self._last_live > info["ale.lives"]:
                t = True
                self._last_live = info["ale.lives"]
                if self._last_live > 0:
                    self._reset_flag = False
                else:
                    self._last_live = 5
                    self._reset_flag = True
            if t:
                terminal = True

        if len(self.previous_frames) > 3:
            self.previous_frames = self.previous_frames[1:] + [state]
        else:
            self.previous_frames += [state]
        state = np.stack(self.previous_frames)
        return state, np.array(np.sum(reward_list) > 0), terminal
