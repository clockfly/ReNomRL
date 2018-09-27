import copy
import numpy as np
from tqdm import tqdm
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore

import renom as rm
from renom.core import Grads
from renom.utility.initializer import Uniform, GlorotUniform
from renom.utility.reinforcement.replaybuffer import ReplayBuffer
from renom_rl.environ import BaseEnv
from renom_rl.noise import OU


import time
try:
    from gym.core import Env as OpenAIEnv
except:
    OpenAIEnv = None


def safe_softplus_ad(x, limit=30):
    mask = x.as_ndarray() > limit
    return mask * x + rm.log(1 + rm.exp(x * ~mask)) * ~mask


def safe_softplus(x, limit=30):
    return np.where(x > limit, x, np.log(1 + np.exp(x)))


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
        for layer in self._network.iter_models():
            if hasattr(layer, "params"):
                layer.params = {}

        self._num_worker = num_worker
        self.envs = [copy.deepcopy(env) for i in range(num_worker)]
        self.loss_func = loss_func
        self._actor_optimizer = optimizer
        self._critic_optimizer = copy.deepcopy(optimizer)
        self.gamma = gamma
        self.semaphore = BoundedSemaphore(1)

        action_shape = env.action_shape
        state_shape = env.state_shape

        if hasattr(action_shape, "__getitem__"):
            self.action_size = action_shape
        else:
            self.action_size = (action_shape,)

        if hasattr(state_shape, "__getitem__"):
            self.state_size = state_shape
        else:
            self.state_size = (state_shape,)

        self.bar = None

        # Dry run.
        with self._network.train():
            loss_p, loss_v = self._network(np.random.rand(1, *self.state_size))
            loss_p = rm.sum(loss_p)
            loss_v = rm.sum(loss_v)

        self._master_critic_variables = loss_v.grad(detach_graph=False)._auto_updates
        self._master_actor_variables = loss_p.grad()._auto_updates
        loss_p = None
        loss_v = None

        self._slave_network = [copy.deepcopy(self._network) for i in range(num_worker)]
        for slave in self._slave_network:
            for layer in list(slave.iter_models()):
                if hasattr(layer, "params") and False:
                    layer.params = {}
            slave(np.random.rand(1, *self.state_size))

    def action(self, state):
        """This method returns an action according to the given state.
        Args:
            state (ndarray): A state of an environment.

        Returns:
            (int, ndarray): Action.
        """
        self._network.set_models(inference=True)
        p = self._network(state.reshape(1, *self.state_size))[0].as_ndarray()
        dim = p.shape[1] // 2
        return np.random.randn() * safe_softplus(p[:, dim:]) + p[:, :dim]

    def fit(self, episode=1000, episode_step=2000, min_greedy=0.0, max_greedy=0.9, greedy_step=1000000, test_step=2000, test_frequency=100, render=False, callback_end_epoch=None):
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

        def run_agent(args):
            network, env, learning_rate = args
            avg_reward_list = []
            for e in range(test_frequency // self._num_worker):
                # Run episode for specified times.
                state = env.reset()
                reward_list = []
                for _ in range(episode_step // tmax):
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
                            network.set_models(inference=True)
                            p = network(state.reshape(1, *self.state_size))[0].as_ndarray()
                            action = np.random.randn() * safe_softplus(p[:, dim:]) + p[:, :dim]
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
                        if terminal:
                            break

                    value = 0 if trajectory[-1][TERMINAL] else network(trajectory[-1][STATE])[
                        1].as_ndarray()

                    actor_grad = None
                    critic_grad = None
                    for t in range(len(trajectory))[::-1]:
                        target = trajectory[t][REWARD] + self.gamma * value
                        target = target.reshape(1, 1)
                        network.set_models(inference=False)
                        with network.train():
                            _, v = network(trajectory[t][PRESTATE])
                            critic_loss = self.loss_func(v, target)
                        grad = critic_loss.grad()
                        if critic_grad is None:
                            critic_grad = grad
                        else:
                            for k in critic_grad._auto_updates:
                                critic_grad.variables[id(k)] += grad.variables[id(k)]

                        with network.train():
                            p, v = network(trajectory[t][PRESTATE])
                            dim = p.shape[1] // 2
                            u = p[:, :dim]
                            s = safe_softplus_ad(p[:, dim:]) ** 2  # This equals to sigm^2.
                            logs = rm.log(s + 1e-10)
                            log_pi2 = np.log(np.pi * 2)
                            actor_loss = 0.5 * rm.sum((-0.5 * (log_pi2 + logs) - ((trajectory[t][ACTION] - u)**2) / (
                                2 * s + 1e-10)) * (target - v.as_ndarray())) - 0.0001 * (0.5 * rm.sum(log_pi2 + logs + 1))
                        grad = actor_loss.grad(-1 * np.ones_like(actor_loss))
                        if actor_grad is None:
                            actor_grad = grad
                        else:
                            for k in actor_grad._auto_updates:
                                actor_grad.variables[id(k)] += grad.variables[id(k)]

                        value = target

                    self.semaphore.acquire()
                    self._actor_optimizer._lr = learning_rate
                    self._critic_optimizer._lr = learning_rate
                    a_grad = Grads()
                    c_grad = Grads()
                    for m, s in zip(self._master_actor_variables, actor_grad._auto_updates):
                        a_grad.variables[id(m)] = actor_grad.variables[id(s)]
                        a_grad._auto_updates.append(m)
                    for m, s in zip(self._master_critic_variables, critic_grad._auto_updates):
                        c_grad.variables[id(m)] = critic_grad.variables[id(s)]
                        c_grad._auto_updates.append(m)
                    a_grad.update(self._actor_optimizer)
                    c_grad.update(self._critic_optimizer)
                    network.copy_params(self._network)
                    self.semaphore.release()
                    learning_rate = learning_rate - (1e-2 - 1e-7) / (episode_step * episode)
                    learning_rate = float(np.clip(learning_rate, 0, 1e-2))
                    if terminal:
                        break

                avg_reward_list.append(np.sum(reward_list))
                self.bar.update(1)
            return np.mean(avg_reward_list), learning_rate

        next_learning_rate = [float(np.exp(np.random.uniform(np.log(1e-7), np.log(1e-2))))
                              for _ in range(self._num_worker)]
        train_reward_list = []
        test_reward_list = []

        for i in range(episode // test_frequency):
            self.bar = tqdm()
            with ThreadPoolExecutor(max_workers=self._num_worker) as exc:
                result = exc.map(run_agent, [(net, env, lr) for net, env, lr in zip(
                    self._slave_network, self.envs, next_learning_rate)])
            ret = []
            next_learning_rate = []
            for r, l in result:
                ret.append(r)
                next_learning_rate.append(l)

            train_reward_list.append(np.mean(ret))
            test_reward_list.append(self.test(render=render))
            self.bar.set_description("{:04d} Average Train reward: {:5.3f} Test reward: {:5.3f}".format(
                i, train_reward_list[-1], test_reward_list[-1]))
            self.bar.update(0)
            self.bar.refresh()
            self.bar.close()
            if callback_end_epoch is not None:
                callback_end_epoch(i, self._network, train_reward_list, test_reward_list)

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
        return float(sum_reward)
