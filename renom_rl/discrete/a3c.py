import copy
import numpy as np
from tqdm import tqdm
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore

import renom as rm
from renom.utility.initializer import Uniform, GlorotUniform
from renom.utility.reinforcement.replaybuffer import ReplayBuffer
from renom_rl.env import BaseEnv
from renom_rl.noise import OU


import time
try:
    from gym.core import Env as OpenAIEnv
except:
    OpenAIEnv = None


class A3C(object):
    """A3C class
    Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel,
    Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra,
    Continuous control with deep reinforcement learning
    https://arxiv.org/abs/1509.02971

    This class provides a reinforcement learning agent including training and testing methods.
    This class only accepts 'Environment' as a object of 'BaseEnv' class or OpenAI-Gym 'Env' class.

    Args:
        env (BaseEnv, openAI-Env): An instance of Environment to be learned.
                        For example, 'Pendulum-v0' environment, has methods reset, step.
                        env.reset() --> resets initial state of environment
                        env.step(action) --> take action value and returns (next_state, reward, terminal, _)
        actor_network (Model): Actor-Network. If it is None, default ANN is created 
                                with [400, 300] hidden layer sizes
        critic_network (Model): basically a Q(s,a) function Network.
        actor_optimizer : Optimizer object for training actor network.
        critic_optimizer : Optimizer object for training actor network.
        loss_func: Loss function for critic network.
        gamma (float): Discount rate.
        tau (float): target_networks update parameter. If this is 0, weight parameters will be copied.
        batch_size (int): mini batch size.
        buffer_size (float, int): The size of replay buffer.
    """

    def __init__(self, env, network, loss_func=rm.mse,
                 optimizer=rm.Rmsprop(0.001, g=0.99, epsilon=1e-10), gamma=0.99, num_worker=8):

        self._network = network
        self._num_worker = num_worker

        self.envs = [copy.deepcopy(env) for i in range(num_worker)]
        self.loss_func = loss_func
        self._actor_optimizer = optimizer
        self._critic_optimizer = copy.deepcopy(optimizer)
        self.gamma = gamma
        self.semaphore = BoundedSemaphore(1)

        action_shape = env.action_shape
        state_shape = env.state_shape

        self.action_size = action_shape
        self.state_size = state_shape
        self.bar = None

    def action(self, state):
        """This method returns an action according to the given state.
        Args:
            state (ndarray): A state of an environment.

        Returns:
            (int, ndarray): Action.
        """
        self._network.set_models(inference=True)
        p = self._network(state.reshape(1, *self.state_size))[0].as_ndarray()[0]
        choice = np.random.choice(range(4), p=p)
        return choice

    def fit(self, episode=1000, episode_step=2000, min_greedy=0.0, max_greedy=0.9, greedy_step=1000000, test_step=2000, test_frequency=100):
        """ This method executes training of an actor-network.
        Here, target actor & critic network weights are updated after every actor & critic update using self.tau
        Args:
            episode (int): training number of episodes
            episode_step (int): Depends on the type of Environment in-built setting.
                             Environment reaches terminal situation in two cases.
                            (i) In the type of Game, it is game over
                            (ii) Maximum time steps to play                    
        Returns:
            (dict): A dictionary which includes reward list of training and loss list.
        """
        tmax = 5

        self.greedy = min_greedy
        g_step = (max_greedy - min_greedy) / greedy_step

        train_reward_list = []

        def run_agent(args):
            env, learning_rate = args
            avg_reward_list = []
            for e in range(test_frequency//self._num_worker):
                # Run episode for specified times.
                state = env.reset()
                reward_list = []
                for _ in range(episode_step//tmax):
                    # 1 episode.
                    PRESTATE = 0
                    ACTION = 1
                    REWARD = 2
                    STATE = 3
                    TERMINAL = 4
                    trajectory = []
                    for t in range(tmax):
                        # Play until tmax.
                        state = state.reshape(1, *self.state_size)
                        if self.greedy > np.random.rand():
                            action = self.action(state)
                        else:
                            action = env.sample()
                        if isinstance(env, BaseEnv):
                            next_state, reward, terminal = env.step(action)
                        elif isinstance(env, OpenAIEnv):
                            next_state, reward, terminal, _ = env.step(action)
                        next_state = next_state.reshape(1, *self.state_size)
                        trajectory.append((state, action, reward, next_state, terminal))
                        state = next_state
                        reward_list.append(reward)
                        self.greedy += g_step
                        self.greedy = np.clip(self.greedy, min_greedy, max_greedy)
                        if terminal:
                            break
    
                    self.semaphore.acquire()
                    value = 0 if trajectory[-1][TERMINAL] else self._network(trajectory[-1][STATE])[1].as_ndarray()
                    self.semaphore.release()
    
                    actor_grad = None
                    critic_grad = None
                    for t in range(len(trajectory))[::-1]:
                        target = trajectory[t][REWARD] + self.gamma*value
                        target = target.reshape(1, 1)
                        self.semaphore.acquire()
                        self._network.set_models(inference=False)
                        with self._network.train():
                            _, v = self._network(trajectory[t][PRESTATE])
                            critic_loss = self.loss_func(v, target)
                        grad = critic_loss.grad()
                        self.semaphore.release()
                        if critic_grad is None:
                            critic_grad = grad
                        else:
                            for k in critic_grad._auto_updates:
                                critic_grad.variables[id(k)] += grad.variables[id(k)]
        
                        self.semaphore.acquire()
                        with self._network.train():
                            p, v = self._network(trajectory[t][PRESTATE])
                            p = p[:, int(trajectory[t][ACTION])]
                            actor_loss = 0.5*rm.sum(rm.log(p+1e-10)*(target - v.as_ndarray())) - 0.01*rm.sum(p*rm.log(p + 1e-10))
                        grad = actor_loss.grad(-1*np.ones_like(actor_loss))
                        self.semaphore.release()
                        if actor_grad is None:
                            actor_grad = grad
                        else:
                            for k in actor_grad._auto_updates:
                                actor_grad.variables[id(k)] += grad.variables[id(k)]

                        value = target
                    self.semaphore.acquire()
                    self._actor_optimizer._lr = learning_rate
                    self._critic_optimizer._lr = learning_rate
                    actor_grad.update(self._actor_optimizer)
                    critic_grad.update(self._critic_optimizer)
                    self.semaphore.release()
                    learning_rate = learning_rate - (1e-2 - 1e-7)/(episode_step*episode)
                    learning_rate = float(np.clip(learning_rate, 0, 1e-2))
                    if terminal:
                        break

                avg_reward_list.append(np.sum(reward_list))
                self.bar.update(1)
            return np.mean(avg_reward_list), learning_rate

        next_learning_rate = [float(np.exp(np.random.uniform(np.log(1e-7), np.log(1e-2)))) for _ in range(self._num_worker)]
        for i in range(episode//test_frequency):
            self.bar = tqdm()
            with ThreadPoolExecutor(max_workers=self._num_worker) as exc:
                result = exc.map(run_agent, [(e, lr) for e, lr in zip(self.envs, next_learning_rate)])
            ret = []
            next_learning_rate = []
            for r, l in result:
                ret.append(r)
                next_learning_rate.append(l)
                
            self.bar.set_description("Average Train reward: {:5.3f} Test reward: {:5.3f}".format(np.mean(ret), self.test(render=True)))
            self.bar.update(0)
            self.bar.refresh()
            self.bar.close()
                
            
    def test(self, test_steps=2000, render=False):
        '''test the trained network
        Args:
        Return:
            (list): A list of cumulative test rewards
        '''
        env = self.envs[0]
        sum_reward = 0
        state = env.reset()
        for _ in range(test_steps):
            action = self.action(state.reshape(1, *self.state_size))
            if isinstance(env, BaseEnv):
                state, reward, terminal = env.step(action)
            else:
                state, reward, terminal, _ = env.step(action)
            sum_reward += reward
            if render:
                env.render()
            if terminal:
                break
        return sum_reward
