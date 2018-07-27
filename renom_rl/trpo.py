import numpy as np
import renom as rm


class TRPO(object):
    """
    TRPO(Trust Region Policy Optimization) reinforcement learning class.

    John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel
    Trust Region Policy Optimization
    https://arxiv.org/abs/1502.05477
    """

    def __init__(self, env, poilcy_network, value_network):
        self._env = env
        self._policy_network = poilcy_network
        self._value_network = value_network

    def action(self, state):
        pass

    def update(self):
        pass

    def train(self):
        pass

    def train_value_network(self):
        pass

    def update_policy_network(self):
        pass

    def run_episode(self, num_episode, max_step):
        episodes = []
        for e in range(num_episode):
            eps = {
                "state": [],
                "reward": [],
                "action": []
            }
            state = self._env.reset()
            for step in range(max_step):
                action = self.action(state)
                next_state, reward, terminal = self.action(state)
                eps["state"].append(state)
                eps["action"].append(action)
                eps["reward"].append(reward)
                eps["terminal"].append(terminal)
                state = next_state
                if terminal:
                    break
            episodes.append(eps)
        return episodes

    def test(self, test_step=2000, test_greedy=0.95, render=False):
        pass
