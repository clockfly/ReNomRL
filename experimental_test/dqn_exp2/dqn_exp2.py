import os
import gym
import numpy as np
import renom as rm
import matplotlib.pyplot as plt
from renom.utility.initializer import Gaussian
from renom.cuda import set_cuda_active
from renom_rl.discrete.dqn import DQN
from renom_rl.env import BaseEnv
from gym.core import Env
from PIL import Image
from logging import getLogger, StreamHandler, DEBUG, FileHandler

save_name = os.path.splitext(__file__)[0]

set_cuda_active(True)
env = gym.make('BreakoutNoFrameskip-v4')

logger = getLogger(__name__)
handler = FileHandler("%s.log"%save_name)
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

class CustomEnv(BaseEnv):
    
    def __init__(self, env):
        self.env = env
        self.action_shape = 4
        self.state_shape = (4, 84, 84)
        self.previous_frames = []
        self._reset_flag = True
        self._last_live = 5
        super(CustomEnv, self).__init__()
    
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
        # Confirm that the image is processed correctly.
        # Image.fromarray(np.clip(final_image.reshape(84, 84)*255, 0, 255).astype(np.uint8)).save("test.png")
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


def callback(*args):
    episode, model, train_rew, test_rew, error = args
    logger.debug("episode: {:05d} loss: {:6.4f} train: {} test: {}".format(episode, error[-1], train_rew[-1], test_rew[-1]))
    
custom_env = CustomEnv(env)
q_network = rm.Sequential([rm.Conv2d(32, filter=8, stride=4, ignore_bias=False),
                           rm.Relu(),
                           rm.Conv2d(64, filter=4, stride=2, ignore_bias=False),
                           rm.Relu(),
                           rm.Conv2d(64, filter=3, stride=1, ignore_bias=False),
                           rm.Relu(), 
                           rm.Flatten(), 
                           rm.Dense(512, ignore_bias=False),
                           rm.Relu(),
                           rm.Dense(custom_env.action_shape, ignore_bias=False)])

model = DQN(custom_env, q_network)
result = model.fit(render=True, greedy_step=1000000, random_step=50000, update_period=10000, callback_end_epoch=callback)

q_network.save("%s.h5"%save_name)
