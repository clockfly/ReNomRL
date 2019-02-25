#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import copy
import numpy as np

import renom as rm

from renom_rl.utility.replaybuffer import ReplayBuffer
from renom_rl import AgentBase
from renom_rl.environ.env import BaseEnv
from renom_rl.utility.filter import EpsilonSLFilter, EpsilonCFilter, ActionFilter, MaxNodeChooser
from renom_rl.utility.logger import Logger, DQNLogger, AVAILABLE_KEYS

_dqn_keys = AVAILABLE_KEYS["dqn"]["logger"]
_dqn_keys_epoch = AVAILABLE_KEYS["dqn"]["logger_epoch"]


class DQN(AgentBase):
    """DQN class
    This class provides a reinforcement learning agent including training method.

    Args:
        env (BaseEnv): Environment. This must be a child class of ``BaseEnv``.
        q_network (Model): Agent. Q-Network.
        loss_func (function): Loss function for train q-network. Default is ``ClippedMeanSquaredError()``.
        optimizer: Optimizer for train q-network. Default is ``Rmsprop(lr=0.00025, g=0.95)``.
        gamma (float): Discount rate.
        buffer_size (float, int): The size of replay buffer.

    Example:
        >>> import renom as rm
        >>> from renom_rl.discrete.dqn import DQN
        >>> from renom_rl.environ.openai import CartPole00
        >>> model = rm.Sequential(...)
        >>> agent = DQN(
        ...       CartPole00(),
        ...       model,
        ...       loss_func=rm.ClippedMeanSquaredError(),
        ...       buffer_size=1e6
        ...   )
        >>> agent.fit(epoch=10, epoch_step=1)
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
                 optimizer=None, gamma=0.99, buffer_size=1e6,
                 node_selector=None, test_node_selector=None, logger=None):
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
        self.epsilon_update = None

        self.node_selector = MaxNodeChooser() if node_selector is None else node_selector
        self.test_node_selector = MaxNodeChooser() if test_node_selector is None else test_node_selector

        # Check Env class type.
        if isinstance(env, BaseEnv):
            action_shape = env.action_shape
            state_shape = env.state_shape
        else:
            raise Exception("Argument env must be a object of BaseEnv class.")

        # Check state and action shape
        assert state_shape == self.env.reset().shape, \
            "Expected state shape is {} but acctual is {}".format(
                state_shape, self.env.reset().shape)

        action = self._q_network(np.zeros((1, *state_shape))).as_ndarray()
        assert action.shape[1:] == action_shape, \
            "Expected action shape is {} but acctual is {}".format(action_shape, action.shape[1:])

        self._action_shape = action_shape
        self._state_shape = state_shape
        self._buffer = ReplayBuffer([1, ], self._state_shape, buffer_size)
        self._initialize()

        # logger
        logger = DQNLogger() if logger is None else logger
        assert isinstance(logger, Logger), "Argument logger must be Logger class"
        logger._key_check(log_key=_dqn_keys, log_key_epoch=_dqn_keys_epoch)
        self.logger = logger

    def _initialize(self):
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

    def _action(self, state, return_q=False):
        """This method returns an action according to the given state.
        """
        self._q_network.set_models(inference=True)
        act = self._q_network(state[None, ...])
        if return_q:
            return self.node_selector(act), act.as_ndarray()
        return self.node_selector(act)

    def _test_action(self, state):
        """This method returns an action according to the given state.
        """
        self._q_network.set_models(inference=True)
        act = self._q_network(state[None, ...])
        return self.test_node_selector(act)

    def _walk_model(self, model):
        yield self
        for k, v in sorted(self.__dict__.items(), key=lambda x: x[0]):
            if isinstance(v, rm.Model):
                yield self._walk_model(v)

    def _rec_copy(self, obj1, obj2):
        """This function copies the batch normalization parameters"""
        for item_keys in obj1.__dict__.keys():
            if isinstance(obj1.__dict__[item_keys], rm.BatchNormalize):
                obj1.__dict__[item_keys]._mov_mean = obj2.__dict__[item_keys]._mov_mean
                obj1.__dict__[item_keys]._mov_std = obj2.__dict__[item_keys]._mov_std
            elif isinstance(obj1.__dict__[item_keys], rm.Model):
                self._rec_copy(obj1.__dict__[item_keys], obj2.__dict__[item_keys])

    def _update(self):
        """This function updates target network."""
        ## A(B) Copy B to A.
        self._target_q_network.copy_params(self._best_q_network)
        self._rec_copy(self._target_q_network, self._best_q_network)

    def _update_best_q_network(self):
        """This function updates best network in each target update period."""
        self._best_q_network.copy_params(self._q_network)
        self._rec_copy(self._best_q_network, self._q_network)

    def fit(self, epoch=500, epoch_step=250000, batch_size=32, random_step=50000,
            test_step=None, update_period=10000, train_frequency=4,
            action_filter=None):
        """This method executes training of a q-network.
        Training will be done with epsilon-greedy method(default).

        Args:
            epoch (int): Number of epoch for training.
            epoch_step (int): Number of step of one epoch.
            batch_size (int): Batch size.
            random_step (int): Number of random step which will be executed before training.
            test_step (int): Number of test step.
            update_period (int): Period of updating target network.
            train_frequency (int): For the learning step, training is done at this cycle
            action_filter (ActionFilter): Exploration filter during learning. Default is `EpsilonGreedyFilter`.

        Example:
            >>> import renom as rm
            >>> from renom_rl.discrete.dqn import DQN
            >>> from renom_rl.environ.openai import CartPole00
            >>>
            >>> q_network = rm.Sequential([
            ...    # Define network here.
            ... ])
            >>> model = DQN(CartPole00(), q_network)
            >>>
            >>> model.fit()
            epoch 001 avg_loss:0.0031 total reward in epoch: [train:109.000 test: 3.0] avg reward in episode:[train:0.235 test:0.039] e-greedy:0.900: 100%|██████████| 10000/10000 [05:48<00:00, 28.73it/s]
            epoch 002 avg_loss:0.0006 total reward in epoch: [train:116.000 test:14.0] avg reward in episode:[train:0.284 test:0.163] e-greedy:0.900: 100%|██████████| 10000/10000 [05:53<00:00, 25.70it/s]
            ...

        """

        # action filter is set, if not exist then make an instance
        action_filter = action_filter if action_filter is not None else EpsilonSLFilter(
            epsilon_step=int(0.8 * epoch * epoch_step))

        # check
        assert isinstance(
            action_filter, ActionFilter), "action_filter must be a class of ActionFilter"

        assert isinstance(self.logger, Logger), "logger must be Logger class"
        self.logger._key_check(log_key=_dqn_keys, log_key_epoch=_dqn_keys_epoch)

        # random step phase
        print("Run random {} step for storing experiences".format(random_step))

        state = self.env.reset()

        # env start(after reset)
        self.env.start()

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

        count = 0  # update period
        step_count = 0  # steps
        episode_count = 0  # episodes

        # 1 epoch stores multiple epoch steps thus 1 epoch can hold multiple episodes
        for e in range(1, epoch + 1):
            continuous_step = 0
            continuous_step_log = 0
            sum_reward = 0
            sum_reward_log = 0
            nth_episode = 0
            episode_q = []
            episode_loss = []
            
            self.logger.start(epoch_step)

            # env epoch
            self.env.epoch()

            state = self.env.reset()
            loss = 0

            for j in range(epoch_step):

                # set action
                act, q = self._action(state, return_q=True)
                episode_q.append(np.max(q))
                action = action_filter(act, self.env.sample(),
                                       step=step_count, episode=episode_count, epoch=e)
                greedy = action_filter.value()

                # pass it to env
                next_state, reward, terminal = self.env.step(action)

                self._buffer.store(state, np.array(action),
                                   np.array(reward), next_state, np.array(terminal))

               # env epoch step
                self.env.epoch_step()

                sum_reward += reward

                if j % train_frequency == 0 and j:
                    if len(self._buffer) > batch_size:
                        train_prestate, train_action, train_reward, train_state, train_terminal = \
                            self._buffer.get_minibatch(batch_size)

                        # getting q values as target reference
                        self._q_network.set_models(inference=True)
                        self._target_q_network.set_models(inference=True)

                        target = self._q_network(train_prestate).as_ndarray()

                        target.setflags(write=True)
                        max_q_action = np.argmax(self._q_network(train_state).as_ndarray(), axis=1)
                        value = np.amax(self._target_q_network(train_state).as_ndarray(),
                                        axis=1, keepdims=True) * self._gamma * (~train_terminal[:, None])

                        # getting target value
                        for i in range(batch_size):
                            a = train_action[i, 0].astype(np.integer)
                            target[i, a] = train_reward[i] + value[i]

                        # train
                        self._q_network.set_models(inference=False)
                        with self._q_network.train():
                            z = self._q_network(train_prestate)
                            ls = self.loss_func(z, target)
                        ls.grad().update(self._optimizer)
                        loss = np.sum(ls.as_ndarray())
                        episode_loss.append(ls.as_ndarray())
                        # train_loss += loss

                if count % update_period == 0 and count:
                    max_reward_in_each_update_period = -np.Inf
                    self._update()
                    count = 0
                count += 1

                # terminal reset
                if terminal:
                    if max_reward_in_each_update_period <= sum_reward:
                        self._update_best_q_network()
                        max_reward_in_each_update_period = sum_reward

                    self.logger.logger_episode(
                      nth=episode_count,
                      mean_q=np.mean(episode_q),
                      mean_loss=np.mean(episode_loss),
                      cum_reward=sum_reward,
                      model=self._q_network,
                    )
                    # train_sum_rewards_in_each_episode.append(sum_reward)
                    # hold log values
                    sum_reward_log = sum_reward
                    continuous_step_log = continuous_step
                    # reset log values
                    sum_reward = 0
                    continuous_step = 0
                    episode_q = []
                    episode_loss = []
                    # increment episode values
                    nth_episode += 1
                    episode_count += 1

                    self.env.reset()

                self.logger.update(1)
                self.logger.logger(state=state, action=action, reward=reward,
                                   terminal=terminal, next_state=next_state,
                                   total_step=step_count, epoch_step=j, max_step=epoch_step,
                                   total_episode=episode_count, epoch_episode=nth_episode, steps_per_episode=continuous_step_log,
                                   epoch=e, max_epoch=epoch, loss=loss,
                                   sum_reward=sum_reward_log, epsilon=greedy)
                # self.logger.update(1)

                continuous_step += 1
                step_count += 1
                state = next_state

                # if terminate executes, then do execute "continue"
                if self.env.terminate():
                    print("terminated")
                    break

            else:
                summed_test_reward = self.test(test_step, action_filter)
                self.logger.logger_epoch(total_episode=episode_count, epoch_episode=nth_episode,
                                         epoch=e, max_epoch=epoch, test_reward=summed_test_reward, epsilon=greedy)
                self.logger.close()
                continue

            break

    def test(self, test_step=None, action_filter=None, **kwargs):
        """
        Test the trained agent.

        Args:
            test_step (int, None): Number of steps (not episodes) for test. If None is given, this method tests execute only 1 episode.
            action_filter (ActionFilter): Exploartion filter during test. Default is `ConstantFilter(threshold=1.0)`.

        Returns:
            Sum of rewards.
        """

        # if filter_obj argument was specified, the change the object
        action_filter = action_filter if action_filter is not None else EpsilonCFilter()

        assert isinstance(action_filter, ActionFilter)

        sum_reward = 0
        self.env.test_start()
        state = self.env.reset()

        if test_step is None:
            while True:
                action = action_filter.test(self._test_action(state), self.env.sample())

                state, reward, terminal = self.env.step(action)

                sum_reward += float(reward)

                self.env.test_epoch_step()

                if terminal:
                    break

        else:
            for j in range(test_step):
                action = action_filter.test(self._test_action(state), self.env.sample())

                state, reward, terminal = self.env.step(action)

                sum_reward += float(reward)

                self.env.test_epoch_step()

                if terminal:
                    self.env.reset()

        self.env.test_close()
        return sum_reward
