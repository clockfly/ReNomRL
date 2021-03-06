#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import numpy as np

import renom as rm

from renom_rl.utility.replaybuffer import ReplayBuffer
from renom_rl import AgentBase
from renom_rl.environ import BaseEnv
from renom_rl.utility.filter import EpsilonSL, ActionNoiseFilter, OUFilter, NoNoiseFilter
from renom_rl.utility.logger import Logger, DDPGLogger, AVAILABLE_KEYS

_ddpg_keys = AVAILABLE_KEYS["ddpg"]["logger"]
_ddpg_keys_epoch = AVAILABLE_KEYS["ddpg"]["logger_epoch"]


class DDPG(AgentBase):
    """DDPG class

    This class provides a reinforcement learning for DDPG.

    Args:
        env (BaseEnv): Environment. This must be a child class of ``BaseEnv``.
        actor_network (Model): Actor-Network.
        critic_network (Model): Critic-Network. This is a Q(s,a) Network. Requires state and action input.
        loss_func: Loss function for critic network. Default is ``MeanSquaredError()``
        actor_optimizer : Optimizer object for training actor network. Default is ``Adam(lr=0.0001)``
        critic_optimizer : Optimizer object for training actor network. Default is ``Adam(lr=0.001)``
        gamma (float): Discount rate.
        tau (float): target_networks update parameter. If this is 0, weight parameters will be copied.
        buffer_size (float, int): The size of replay buffer.

    Example:
        >>> import renom as rm
        >>> from renom_rl.continuous.ddpg import DDPG
        >>> from renom_rl.environ.openai import Pendulum
        >>>
        >>> class Critic(rm.Model):
        ...
        ...     def __init__(self):
        ...         self.l1 = rm.Dense(2)
        ...         self.l2 = rm.Dense(1)
        ...
        ...     def forward(self, state, action):
        ...         h = rm.concat(self.l1(state), action)
        ...         return self.l2(rm.relu(h))
        ...
        >>> actor = rm.Sequential(...)
        >>> critic = Critic()
        >>> agent = DDPG(
        ...       Pendulum(),
        ...       actor,
        ...       critic,
        ...       loss_func=rm.ClippedMeanSquaredError(),
        ...       buffer_size=1e6
        ...   )
        >>> agent.fit(episode=10000)
        episode 001 avg_loss: 0.004 total_reward [train:2.000 test:-] e-greedy:0.000: : 190it [00:03, 48.42it/s]
        episode 002 avg_loss: 0.003 total_reward [train:0.000 test:-] e-greedy:0.000: : 126it [00:02, 50.59it/s]
        episode 003 avg_loss: 0.003 total_reward [train:3.000 test:-] e-greedy:0.001: : 250it [00:04, 51.31it/s]

    Reference:
        | Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel,
        | Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra,
        | Continuous control with deep reinforcement learning
        | https://arxiv.org/abs/1509.02971
        |

    """

    def __init__(self, env, actor_network, critic_network, loss_func=None,
                 actor_optimizer=None, critic_optimizer=None, gamma=0.99,
                 tau=0.001, buffer_size=1e6, logger=None):
        super(DDPG, self).__init__()
        if loss_func is None:
            loss_func = rm.MeanSquaredError()
        if actor_optimizer is None:
            actor_optimizer = rm.Adam(0.0001)
        if critic_optimizer is None:
            critic_optimizer = rm.Adam(0.001)

        self._actor = actor_network
        self._target_actor = copy.deepcopy(self._actor)
        self._critic = critic_network
        self._target_critic = copy.deepcopy(self._critic)
        self.env = env
        self.loss_func = loss_func
        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.tau = tau

        if isinstance(env, BaseEnv):
            action_shape = env.action_shape
            state_shape = env.state_shape
            if not hasattr(action_shape, "__getitem__"):
                action_shape = (action_shape, )
            if not hasattr(state_shape, "__getitem__"):
                state_shape = (state_shape, )
        else:
            raise Exception("Argument env must be a object of BaseEnv class.")

        # Check env object
        # Check sample method.
        if isinstance(env, BaseEnv):
            sample = self.env.sample()
        else:
            raise Exception("Argument env must be a object of BaseEnv class.")

        assert isinstance(sample, np.ndarray), \
            "Sampled action from env object must be numpy ndarray. Actual is {}".format(
                type(sample))

        # Check state and action shape
        assert state_shape == self.env.reset().shape, \
            "Expected state shape is {}. Actual is {}.".format(state_shape, self.env.reset().shape)
        action_sample = self._actor(np.zeros((1, *state_shape))).as_ndarray()
        assert action_sample.shape[1:] == action_shape, \
            "Expected state shape is {}. Actual is {}.".format(
                action_shape, action_sample.shape[1:])
        #####
        self.action_size = action_shape
        self.state_size = state_shape
        self._buffer = ReplayBuffer(self.action_size, self.state_size, buffer_size)
        self._initialize()

        # logger
        logger = DDPGLogger() if logger is None else logger
        assert isinstance(logger, Logger), "Argument logger must be Logger class"
        logger._key_check(log_key=_ddpg_keys, log_key_epoch=_ddpg_keys_epoch)
        self.logger = logger

    def _action(self, state):
        """This method returns an action according to the given state.
        Args:
            state (ndarray): A state of an environment.

        Returns:
            (int, ndarray): Action.
        """
        self._actor.set_models(inference=True)
        return self._actor(state.reshape(1, *self.state_size)).as_ndarray()[0]

    def fit(self, epoch=1000, epoch_step=250000, batch_size=64, random_step=5000,
            test_step=None, train_frequency=1, action_filter=None,
            ):
        """ This method executes training of an actor-network.
        Here, target actor & critic network weights are updated after every actor & critic update using self.tau

        Args:
            epoch (int): Training number of epochs.
            epoch_step (int): Number of step of one epoch.
            batch_size (int): Batch size.
            random_step (int): Number of random step which will be executed before training.
            test_step (int): Number of test step.
            train_frequency (int): For the learning step, training is done at this cycle.
            action_filter (ActionFilter): Exploration filter during learning. Default is ``OUFilter``.

        Example:
            >>> import renom as rm
            >>> from renom_rl.continuous.ddpg import DDPG
            >>> from renom_rl.environ.openai import Pendulum
            >>>
            >>> class Critic(rm.Model):
            ...
            ...     def __init__(self):
            ...         self.l1 = rm.Dense(2)
            ...         self.l2 = rm.Dense(1)
            ...
            ...     def forward(self, state, action):
            ...         h = rm.concat(self.l1(state), action)
            ...         return self.l2(rm.relu(h))
            ...
            >>> actor = rm.Sequential(...)
            >>> critic = Critic()
            >>>
            >>> agent = DDPG(
            ...       Pendulum(),
            ...       actor,
            ...       critic,
            ...       loss_func=rm.ClippedMeanSquaredError(),
            ...       buffer_size=1e6
            ...   )
            >>> @agent.event.end_epoch
            >>> def callback(epoch, ddpg_model, train_rew, test_rew, avg_loss):
            ... # This function will be called end of each epoch.
            ...
            >>>
            >>> agent.fit()
            epoch 001 avg_loss:0.0031 total reward in epoch: [train:109.000 test: 3.0] avg reward in episode:[train:0.235 test:0.039] e-greedy:0.900: 100%|██████████| 10000/10000 [05:48<00:00, 28.73it/s]
            epoch 002 avg_loss:0.0006 total reward in epoch: [train:116.000 test:14.0] avg reward in episode:[train:0.284 test:0.163] e-greedy:0.900: 100%|██████████| 10000/10000 [05:53<00:00, 25.70it/s]
            ...

        """

        _e = EpsilonSL(epsilon_step=int(0.8 * epoch * epoch_step))
        action_filter = action_filter if action_filter is not None else OUFilter(epsilon=_e)

        # check
        assert isinstance(
            action_filter, ActionNoiseFilter), "action_filter must be a class of ActionNoiseFilter"

        assert isinstance(self.logger, Logger), "logger must be Logger class"
        self.logger._key_check(log_key=_ddpg_keys, log_key_epoch=_ddpg_keys_epoch)

        state = self.env.reset()

        for _ in range(1, random_step + 1):
            # if isinstance(self.env, BaseEnv):
            action = self.env.sample()
            next_state, reward, terminal = self.env.step(action)
            # else:
            #     raise Exception("Argument env must be a object of BaseEnv class.")
            self._buffer.store(state, action,
                               np.array(reward), next_state, np.array(terminal))
            state = next_state
            if terminal:
                state = self.env.reset()

        count = 0  # update period
        step_count = 0  # steps
        episode_count = 0  # episodes

        for e in range(1, epoch + 1):
            continuous_step = 0
            continuous_step_log = 0
            sum_reward_episode = 0.0
            sum_reward_episode_log = 0.0
            nth_episode = 0
            self.logger.start(epoch_step)

            # env epoch
            self.env.epoch()

            state = self.env.reset()
            loss = 0.0

            for j in range(epoch_step):

                # set action
                action = action_filter(self._action(state),
                                       step=step_count, episode=episode_count, epoch=e)
                e_rate = action_filter.value()
                noise_value = action_filter.sample()

                # if isinstance(self.env, BaseEnv):
                next_state, reward, terminal = self.env.step(action)

                self._buffer.store(state, action,
                                   np.array(reward), next_state, np.array(terminal))

                if len(self._buffer) > batch_size and j % train_frequency == 0:
                    train_prestate, train_action, train_reward, train_state, train_terminal = \
                        self._buffer.get_minibatch(batch_size)

                    qmax = self._target_value_function(train_state).as_ndarray()

                    target_q = []
                    for k in range(batch_size):
                        if train_terminal[k]:
                            target_q.append(np.asarray([train_reward[k]]))
                        else:
                            rr = train_reward[k] + (self.gamma * qmax[k])
                            target_q.append(rr)

                    target_q = np.asarray(target_q).reshape(batch_size, 1)

                    self._critic.set_models(inference=False)
                    with self._critic.train():
                        value = self._critic(train_prestate, train_action)
                        critic_loss = self.loss_func(value, target_q)

                    self._actor.set_models(inference=False)
                    with self._actor.train(), self._critic.train():
                        actor_loss = self._value_function(train_prestate) / len(train_prestate)
                    target_actor_loss = self._target_value_function(
                        train_prestate) / len(train_prestate)

                    critic_loss.grad().update(self._critic_optimizer)
                    with self._critic.prevent_update():
                        actor_loss.grad(-1 * np.ones_like(actor_loss)).update(self._actor_optimizer)

                    loss += critic_loss.as_ndarray()
                    self._update()

                if terminal:
                    # hold log values
                    sum_reward_episode_log = sum_reward_episode
                    continuous_step_log = continuous_step
                    # reset log values
                    sum_reward_episode = 0.0
                    continuous_step = 0
                    # increment episode values
                    nth_episode += 1
                    episode_count += 1

                    state = self.env.reset()

                self.logger.logger(state=state, action=action, reward=reward,
                                   terminal=terminal, next_state=next_state,
                                   total_step=step_count, epoch_step=j, max_step=epoch_step, steps_per_episode=continuous_step_log,
                                   total_episode=episode_count, epoch_episode=nth_episode,
                                   epoch=e, max_epoch=epoch, loss=loss,
                                   sum_reward=sum_reward_episode_log, epsilon=e_rate, noise_value=noise_value)
                self.logger.update(1)

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
                                         epoch=e, max_epoch=epoch, test_reward=summed_test_reward,
                                         epsilon=e_rate, noise_value=noise_value)
                self.logger.close()
                continue

            break

        # env close
        self.env.close()

    def _value_function(self, state):
        '''Value of predict network Q_predict(s,a)

        Args:
            state: input state

        Returns:
            value: Q(s,a) value
        '''
        action = self._actor(state)
        value = self._critic(state, action)
        return value

    def _target_value_function(self, state):
        '''Value of target network Q_target(s,a).

        Args:
            state: input state

        Returns:
            value: Q(s,a) value
        '''
        action = self._target_actor(state)
        value = self._target_critic(state, action)
        return value

    def _initialize(self):
        '''Target actor and critic networks are initialized with same neural network weights as actor & critic network'''
        for layer in list(self._walk_model(self._target_critic)) + list(self._walk_model(self._target_actor)):
            if hasattr(layer, "params") and False:
                layer.params = {}

        for layer in list(self._walk_model(self._critic)) + list(self._walk_model(self._actor)):
            if hasattr(layer, "params") and False:
                layer.params = {}

        state = np.random.rand(1, *self.state_size)
        action = self._target_actor(state)
        self._target_critic(state, action)
        self._actor.copy_params(self._target_actor)
        self._critic.copy_params(self._target_critic)

    def _walk_model(self, model):
        yield model
        for k, v in sorted(model.__dict__.items(), key=lambda x: x[0]):
            if isinstance(v, rm.Model) and v != self:
                yield from self._walk_model(v)

    def _update(self):
        '''Updare target networks'''
        for ql, tql in zip(self._walk_model(self._actor), self._walk_model(self._target_actor)):
            if not hasattr(ql, 'params'):
                continue
            for k in ql.params.keys():
                tql.params[k] = ql.params[k] * self.tau + tql.params[k] * (1 - self.tau)

        for ql, tql in zip(self._walk_model(self._critic), self._walk_model(self._target_critic)):
            if not hasattr(ql, 'params'):
                continue
            for k in ql.params.keys():
                tql.params[k] = ql.params[k] * self.tau + tql.params[k] * (1 - self.tau)

    def test(self, test_step=None, action_filter=None):
        """
        Test the trained agent.

        Args:
            test_step (int, None): Number of steps for test. If None is given, this method tests just 1 episode.
            action_filter (ActionFilter): Exploration filter during learning. Default is ``ConstantFilter(threshold=1.0)``.

        Returns:
            Sum of rewards.
        """
        # if filter_obj argument was specified, the change the object
        action_filter = action_filter if action_filter is not None else NoNoiseFilter()

        assert isinstance(action_filter, ActionNoiseFilter)

        sum_reward = 0
        self.env.test_start()
        state = self.env.reset()

        if test_step is None:
            while True:
                action = action_filter.test(self._action(state))

                state, reward, terminal = self.env.step(action)

                sum_reward += float(reward)

                self.env.test_epoch_step()

                if terminal:
                    break
        else:
            for j in range(test_step):
                action = action_filter.test(self._action(state))

                state, reward, terminal = self.env.step(action)

                sum_reward += float(reward)

                self.env.test_epoch_step()

                if terminal:
                    self.env.reset()

        self.env.test_close()
        return sum_reward
