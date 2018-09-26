from .env import BaseEnv
import gym


class Pendulum(BaseEnv):

    def __init__(self):
        self.env = gym.make('Pendulum-v0')
        self.action_shape = 1
        self.state_shape = 3

    def reset(self):
        return self.env.reset().reshape(3)

    def sample(self):
        return self.env.action_space.sample()

    def step(self, action):
        state, reward, terminal = self.env.step(action)[:3]
        return state.reshape(3), reward, terminal

    def render(self):
        self.env.render()
