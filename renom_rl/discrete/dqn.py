#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import copy
import numpy as np
from tqdm import tqdm

import renom as rm
from renom_rl import AgentBase
from renom_rl.environ.env import BaseEnv
from renom_rl.utility.event_handler import EventHandler
from renom_rl.utility.replaybuffer import ReplayBuffer


class DQN(AgentBase):
    """DQN class
    This class provides a reinforcement learning agent including training method.

    Args:
        env (BaseEnv): Environment. This must be a child class of BaseEnv.
        q_network (Model): Q-Network.
        loss_func (function): Loss function for train q-network. rm.ClippedMeanSquaredError().
        optimizer: Optimizer for train q-network. Default is Rmsprop(lr=0.00025, g=0.95).
        gamma (float): Discount rate.
        buffer_size (float, int): The size of replay buffer.

    Example:
        >>> import renom as rm
        >>> from renom_rl.discrete.dqn import DQN
        >>> from renom_rl.environ.openai import Breakout
        >>> model = rm.Sequential(...)
        >>> agent = DQN(
        ...       Breakout(),
        ...       model,
        ...       loss_func=rm.ClippedMeanSquaredError(),
        ...       buffer_size=1e6
        ...   )
        >>> agent.train(episode=10000)
        episode 001 avg_loss: 0.004 total_reward [train:2.000 test:-] e-greedy:0.000: : 190it [00:03, 48.42it/s]
        episode 002 avg_loss: 0.003 total_reward [train:0.000 test:-] e-greedy:0.000: : 126it [00:02, 50.59it/s]
        episode 003 avg_loss: 0.003 total_reward [train:3.000 test:-] e-greedy:0.001: : 250it [00:04, 51.31it/s]
        ...

    References:
        | Volodymyr Mnih Koray Kavukcuoglu David Silver Alex Graves Ioannis Antonoglou Daan Wierstra Martin Riedmille
        | **Playing Atari with Deep Reinforcement Learning**
        | https://arxiv.org/abs/1312.5602
        |


    """

    def __init__(self, env, q_network, loss_func=None,
                 optimizer=None, gamma=0.99, buffer_size=1e6):
        super(DQN, self).__init__()

        if loss_func is None:
            loss_func = rm.ClippedMeanSquaredError()

        if optimizer is None:
            optimizer = rm.Rmsprop(lr=0.00025, g=0.95)

        # Reset parameters.
        self._q_network = q_network

        # Copy network architectures.
        # Target network.
        self._target_q_network = copy.deepcopy(self._q_network)
        # The network which earned highest summed reward in each update period.
        self._best_q_network = copy.deepcopy(self._q_network)

        self._gamma = gamma
        self.env = env
        self.loss_func = loss_func
        self._optimizer = optimizer
        self.gamma = gamma

        # Check Env class type.
        if isinstance(env, BaseEnv):
            action_shape = env.action_shape
            state_shape = env.state_shape
        else:
            raise Exception("Argument env must be a object of BaseEnv class.")

        # Check state and action shape
        assert state_shape == self.env.reset().shape, \
            "Expected state shape is {} but accual is {}".format(
                state_shape, self.env.reset().shape)

        action = self._q_network(np.zeros((1, *state_shape))).as_ndarray()
        assert action.shape[1:] == action_shape, \
            "Expected action shape is {} but accual is {}".format(action_shape, action.shape[1:])

        self._action_shape = action_shape
        self._state_shape = state_shape
        self._buffer = ReplayBuffer([1, ], self._state_shape, buffer_size)
        self.events = EventHandler()
        self.initialize()

    def initialize(self):
        '''Target q-network is initialized with same neural network weights of q-network.'''
        # Reset weight.
        for layer in self._q_network.iter_models():
            if hasattr(layer, "params"):
                layer.params = {}

        for layer in list(self._target_q_network.iter_models()):
            if hasattr(layer, "params"):
                layer.params = {}

        # Initiallize weight.
        self._target_q_network(np.random.randn(1, *self._state_shape))
        self._q_network.copy_params(self._target_q_network)

    def action(self, state):
        """This method returns an action according to the given state.

        Args:
            state (ndarray): A state of an environment.

        Returns:
            (int, ndarray): Action.

        """
        self._q_network.set_models(inference=True)
        return np.argmax(self._q_network(state[None, ...]).as_ndarray(), axis=1)

    def _walk_model(self, model):
        yield self
        for k, v in sorted(self.__dict__.items(), key=lambda x: x[0]):
            if isinstance(v, rm.Model):
                yield self._walk_model(v)

    def _rec_copy(self, obj1, obj2):
        for item_keys in obj1.__dict__.keys():
            if isinstance(obj1.__dict__[item_keys], rm.BatchNormalize):
                obj1.__dict__[item_keys]._mov_mean = obj2.__dict__[item_keys]._mov_mean
                obj1.__dict__[item_keys]._mov_std = obj2.__dict__[item_keys]._mov_std
            elif isinstance(obj1.__dict__[item_keys], rm.Model):
                self._rec_copy(obj1.__dict__[item_keys], obj2.__dict__[item_keys])

    def update(self):
        """This function updates target network."""
        self._target_q_network.copy_params(self._best_q_network)
        self._rec_copy(self._target_q_network, self._best_q_network)

    def update_best_q_network(self):
        """This function updates best network in each epoch."""
        self._best_q_network.copy_params(self._q_network)
        self._rec_copy(self._best_q_network, self._q_network)

    def fit(self, epoch=500, epoch_step=250000, batch_size=32, random_step=50000,
            test_step=2000, update_period=10000, train_frequency=4, min_greedy=0.0,
            max_greedy=0.9, greedy_step=1000000, test_greedy=0.95, render=False, callback_end_epoch=None):
        """This method executes training of a q-network.
        Training will be done with epsilon-greedy method.

        You can define following callback functions.

        - end_epoch
            Args:
                epoch (int):
                model (DQN):
                summed_train_reward_in_current_epoch (float):
                summed_test_reward_in_current_epoch (float):
                average_train_lossin_current_epoch (float):

        Args:
            epoch (int): Number of epoch for training.
            epoch_step (int): Number of step of one epoch.
            batch_size (int): Batch size.
            random_step (int): Number of random step which will be executed before training.
            test_step (int): Number of test step.
            update_period (int): Period of updating target network.
            train_frequency (int): For the learning step, training is done at this cycle
            min_greedy (int): Minimum greedy value
            max_greedy (int): Maximum greedy value
            greedy_step (int): Number of step
            test_greedy (int): Greedy threshold
            render (bool): If True is given, BaseEnv.render() method will be called in test time.

        Example:
            >>> import renom as rm
            >>> from renom_rl.discrete.dqn import DQN
            >>> from renom_rl.environ.openai import Breakout
            >>>
            >>> q_network = rm.Sequential([
            ...    # Define network here.
            ... ])
            >>> model = DQN(Breakout(), q_network)
            >>> 
            >>> @model.event.end_epoch
            >>> def callback(epoch, ddqn, train_rew, test_rew, avg_loss):
            ... # This function will be called end of each epoch. 
            ... 
            >>> 
            >>> model.fit()
            epoch 001 avg_loss:0.0031 total reward in epoch: [train:109.000 test: 3.0] avg reward in episode:[train:0.235 test:0.039] e-greedy:0.900: 100%|██████████| 10000/10000 [05:48<00:00, 28.73it/s]
            epoch 002 avg_loss:0.0006 total reward in epoch: [train:116.000 test:14.0] avg reward in episode:[train:0.284 test:0.163] e-greedy:0.900: 100%|██████████| 10000/10000 [05:53<00:00, 25.70it/s]
            ...

        """
        greedy = min_greedy
        g_step = (max_greedy - min_greedy) / greedy_step

        print("Run random {} step for storing experiences".format(random_step))

        state = self.env.reset()
        for i in range(1, random_step + 1):
            action = self.env.sample()
            next_state, reward, terminal = self.env.step(action)

            self._buffer.store(state, np.array(action),
                               np.array(reward), next_state, np.array(terminal))
            state = next_state
            if terminal:
                state = self.env.reset()

        # History of Learning
        max_reward_in_each_update_period = -np.Inf

        count = 0
        for e in range(1, epoch + 1):
            sum_reward = 0
            train_loss = 0
            nth_episode = 0
            train_sum_rewards_in_each_episode = []
            tq = tqdm(range(epoch_step))
            state = self.env.reset()

            for j in range(epoch_step):
                if greedy > np.random.rand():  # and state is not None:
                    self._q_network.set_models(inference=True)
                    action = np.argmax(np.atleast_2d(self._q_network(
                        state[None, ...]).as_ndarray()), axis=1)
                else:
                    action = self.env.sample()

                next_state, reward, terminal = self.env.step(action)

                greedy += g_step
                greedy = np.clip(greedy, min_greedy, max_greedy)
                self._buffer.store(state, np.array(action),
                                   np.array(reward), next_state, np.array(terminal))

                sum_reward += reward
                state = next_state

                if len(self._buffer) > batch_size:
                    train_prestate, train_action, train_reward, train_state, train_terminal = \
                        self._buffer.get_minibatch(batch_size)

                    self._q_network.set_models(inference=True)
                    self._target_q_network.set_models(inference=True)

                    target = self._q_network(train_prestate).as_ndarray()

                    target.setflags(write=True)
                    max_q_action = np.argmax(self._q_network(train_state).as_ndarray(), axis=1)
                    value = np.amax(self._target_q_network(train_state).as_ndarray(),
                                    axis=1, keepdims=True) * self._gamma * (~train_terminal[:, None])

                    for i in range(batch_size):
                        a = train_action[i, 0].astype(np.integer)
                        target[i, a] = train_reward[i] + value[i]

                    self._q_network.set_models(inference=False)
                    with self._q_network.train():
                        z = self._q_network(train_prestate)
                        ls = self.loss_func(z, target)
                    ls.grad().update(self._optimizer)
                    loss = np.sum(ls.as_ndarray())
                    train_loss += loss

                    if count % update_period == 0 and count:
                        max_reward_in_each_update_period = -np.Inf
                        self.update()
                        count = 0
                    count += 1

                if terminal:
                    if max_reward_in_each_update_period <= sum_reward:
                        self.update_best_q_network()
                        max_reward_in_each_update_period = sum_reward
                    train_sum_rewards_in_each_episode.append(sum_reward)
                    sum_reward = 0
                    nth_episode += 1
                    self.env.reset()

                msg = "epoch {:04d} loss {:5.4f} rewards in epoch {:4.3f} episode {:04d} rewards in episode {:4.3f}."\
                    .format(e, loss, np.sum(train_sum_rewards_in_each_episode) + sum_reward, nth_episode,
                            train_sum_rewards_in_each_episode[-1] if len(train_sum_rewards_in_each_episode) > 0 else 0)

                tq.set_description(msg)
                tq.update(1)

            # Calc
            avg_error = train_loss / (j + 1)
            avg_train_reward = np.mean(train_sum_rewards_in_each_episode)
            summed_train_reward = np.sum(train_sum_rewards_in_each_episode) + sum_reward
            summed_test_reward = self.test(test_step, test_greedy, render)

            self._append_history(e, avg_error, avg_train_reward,
                                 summed_train_reward, summed_test_reward)

            msg = "epoch {:03d} avg_loss:{:6.4f} total reward in epoch: [train:{:4.3f} test:{:4.3}] " + \
                "avg train reward in episode:{:4.3f} e-greedy:{:4.3f}"
            msg = msg.format(e, avg_error, summed_train_reward,
                             summed_test_reward, avg_train_reward, greedy)

            self.events.on("end_epoch", e, self, avg_error, avg_train_reward,
                           summed_train_reward, summed_test_reward)

            tq.set_description(msg)
            tq.update(0)
            tq.refresh()
            tq.close()

    def test(self, test_step=None, test_greedy=0.95, render=False):
        """
        Test the trained agent.

        Args:
            test_step (int, None): Number of steps for test. If None is given, this method tests just 1 episode.
            test_greedy (float): Greedy ratio of action.
            render (bool): If True is given, BaseEnv.render() method will be called.

        Returns:
            (int): Sum of rewards.
        """
        sum_reward = 0
        state = self.env.reset()

        if test_step is None:
            while True:
                if test_greedy > np.random.rand():
                    action = self.action(state)
                else:
                    action = self.env.sample()

                state, reward, terminal = self.env.step(action)

                sum_reward += float(reward)

                if render:
                    self.env.render()

                if terminal:
                    break
        else:
            for j in range(test_step):
                if test_greedy > np.random.rand():
                    action = self.action(state)
                else:
                    action = self.env.sample()

                state, reward, terminal = self.env.step(action)

                sum_reward += float(reward)

                if render:
                    self.env.render()

                if terminal:
                    self.env.reset()

        return sum_reward
