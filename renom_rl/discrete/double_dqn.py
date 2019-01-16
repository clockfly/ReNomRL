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
from renom_rl.utility.filter import EpsilonSLFilter, EpsilonCFilter, ActionFilter, MaxNodeChooser


class DoubleDQN(AgentBase):
    """Double DQN class
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
        >>> from renom_rl.discrete.double_dqn import DoubleDQN
        >>> from renom_rl.environ.openai import Breakout
        >>> model = rm.Sequential(...)
        >>> agent = DQN(
        ...       Breakout(),
        ...       model,
        ...       loss_func=rm.ClippedMeanSquaredError(),
        ...       buffer_size=1e6
        ...   )
        >>> agent.fit(episode=10000)
        episode 001 avg_loss: 0.004 total_reward [train:2.000 test:-] e-greedy:0.000: : 190it [00:03, 48.42it/s]
        episode 002 avg_loss: 0.003 total_reward [train:0.000 test:-] e-greedy:0.000: : 126it [00:02, 50.59it/s]
        episode 003 avg_loss: 0.003 total_reward [train:3.000 test:-] e-greedy:0.001: : 250it [00:04, 51.31it/s]
        ...

    References:
        | Hado van Hasselt, Arthur Guez, David Silver
        | **Deep Reinforcement Learning with Double Q-learning**
        | https://arxiv.org/abs/1509.06461
        |

    """

    def __init__(self, env, q_network, loss_func=None,
                 optimizer=None, gamma=0.99, buffer_size=1e6,
                 node_selector=None, test_node_selector=None):
        super(DoubleDQN, self).__init__()

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
            "Expected state shape is {} but accual is {}".format(
                state_shape, self.env.reset().shape)

        action = self._q_network(np.zeros((1, *state_shape))).as_ndarray()
        assert action.shape[1:] == action_shape, \
            "Expected action shape is {} but accual is {}".format(action_shape, action.shape[1:])

        self._action_shape = action_shape
        self._state_shape = state_shape
        self._buffer = ReplayBuffer([1, ], self._state_shape, buffer_size)
        self.events = EventHandler()
        self._initialize()

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

    def _action(self, state):
        """This method returns an action according to the given state.

        Args:
            state (ndarray): A state of an environment.

        Returns:
            (int, ndarray): Action.

        """
        self._q_network.set_models(inference=True)
        act = self._q_network(state[None, ...])
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
        for item_keys in obj1.__dict__.keys():
            if isinstance(obj1.__dict__[item_keys], rm.BatchNormalize):
                obj1.__dict__[item_keys]._mov_mean = obj2.__dict__[item_keys]._mov_mean
                obj1.__dict__[item_keys]._mov_std = obj2.__dict__[item_keys]._mov_std
            elif isinstance(obj1.__dict__[item_keys], rm.Model):
                self._rec_copy(obj1.__dict__[item_keys], obj2.__dict__[item_keys])

    def _update(self):
        """This function updates target network."""
        self._target_q_network.copy_params(self._best_q_network)
        self._rec_copy(self._target_q_network, self._best_q_network)

    def _update_best_q_network(self):
        """This function updates best network in each epoch."""
        self._best_q_network.copy_params(self._q_network)
        self._rec_copy(self._best_q_network, self._q_network)

    def fit(self, epoch=500, epoch_step=250000, batch_size=32, random_step=50000,
            test_step=None, update_period=10000, train_frequency=4,
            action_filter=None, callback_end_epoch=None):
        """This method executes training of a q-network.
        Training will be done with epsilon-greedy method(default).

        Args:
            epoch (int): Number of epoch for training.
            epoch_step (int): Number of step of one epoch.
            batch_size (int): Batch size.
            random_step (int): Number of random step which will be executed before training.
            test_step (int): Number of test step.
            update_period (int): Period of updating target network.
            train_frequency (int): For the learning step, training is done at this cycle.
            action_filter (ActionFilter): Exploration filter during learning. Default is ``EpsilonGreedyFilter``.


        Example:
            >>> import renom as rm
            >>> from renom_rl.discrete.double_dqn import DoubleDQN
            >>> from renom_rl.environ.openai import Breakout
            >>>
            >>> q_network = rm.Sequential([
            ...    # Define network here.
            ... ])
            >>> model = DoubleDQN(Breakout(), q_network)
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

        action_filter = action_filter if action_filter is not None else EpsilonSLFilter(
            epsilon_step=int(0.8 * epoch * epoch_step))

        assert isinstance(action_filter, ActionFilter)

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
            sum_reward = 0
            train_loss = 0
            nth_episode = 0
            train_sum_rewards_in_each_episode = []
            tq = tqdm(range(epoch_step))

            # env epoch
            self.env.epoch()

            state = self.env.reset()
            loss = 0

            for j in range(epoch_step):
                # if greedy > np.random.rand():  # and state is not None:
                #     self._q_network.set_models(inference=True)
                #     action = np.argmax(np.atleast_2d(self._q_network(
                #         state[None, ...]).as_ndarray()), axis=1)
                # else:
                #     action = self.env.sample()

                # set action
                action = action_filter(self._action(state), self.env.sample(),
                                       step=step_count, episode=episode_count, epoch=e)
                greedy = action_filter.value()

                # pass it to env
                next_state, reward, terminal = self.env.step(action)

                self._buffer.store(state, np.array(action),
                                   np.array(reward), next_state, np.array(terminal))

               # env epoch step
                self.env.epoch_step()

                sum_reward += reward
                state = next_state
                if j % train_frequency == 0 and j:
                    if len(self._buffer) > batch_size:
                        train_prestate, train_action, train_reward, train_state, train_terminal = \
                            self._buffer.get_minibatch(batch_size)

                        # getting q values as target reference
                        self._q_network.set_models(inference=True)
                        self._target_q_network.set_models(inference=True)

                        target = self._q_network(train_prestate).as_ndarray()

                        # dqn feature here
                        target.setflags(write=True)
                        max_q_action = np.argmax(self._q_network(train_state).as_ndarray(), axis=1)
                        value = self._target_q_network(train_state).as_ndarray()[(range(len(train_state)),
                                                                                  max_q_action)][:, None] * self._gamma * (~train_terminal[:, None])

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
                        train_loss += loss

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
                    train_sum_rewards_in_each_episode.append(sum_reward)
                    sum_reward = 0
                    nth_episode += 1
                    episode_count += 1

                    self.env.reset()

                # message print
                msg = "epoch {:04d} epsilon {:.4f} loss {:5.4f} rewards in epoch {:4.3f} episode {:04d} rewards in episode {:4.3f}."\
                    .format(e, greedy, loss, np.sum(train_sum_rewards_in_each_episode) + sum_reward, nth_episode,
                            train_sum_rewards_in_each_episode[-1] if len(train_sum_rewards_in_each_episode) > 0 else 0)
                step_count += 1
                tq.set_description(msg)
                tq.update(1)

                # event handler
                self.events.on("step", e, reward, self, step_count, episode_count, greedy)

                # if terminate executes, then do execute "continue"
                if self.env.terminate():
                    print("terminated")
                    break

            else:
                # Calc
                avg_error = train_loss / (j + 1)
                avg_train_reward = np.mean(train_sum_rewards_in_each_episode)
                summed_train_reward = np.sum(train_sum_rewards_in_each_episode) + sum_reward
                summed_test_reward = self.test(test_step, action_filter)

                self._append_history(e, avg_error, avg_train_reward,
                                     summed_train_reward, summed_test_reward)

                msg = "epoch {:03d} avg_loss:{:6.4f} total reward in epoch: [train:{:4.3f} test:{:4.3}] " + \
                    "avg train reward in episode:{:4.3f} e-greedy:{:4.3f}"
                msg = msg.format(e, avg_error, summed_train_reward,
                                 summed_test_reward, avg_train_reward, greedy)

                self.events.on("end_epoch", e, self, avg_error, avg_train_reward,
                               summed_train_reward, summed_test_reward, greedy)

                tq.set_description(msg)
                tq.update(0)
                tq.refresh()
                tq.close()
                continue

            tq.update(0)
            tq.refresh()
            tq.close()
            break

        # env close
        self.env.close()

    def test(self, test_step=None, action_filter=None, **kwargs):
        """
        Test the trained agent.

        Args:
            test_step (int, None): Number of steps (not episodes) for test. If None is given, this method tests execute only 1 episode.
            action_filter (ActionFilter): Exploartion filter during test. Default is ``ConstantFilter(threshold=1.0)``.

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
