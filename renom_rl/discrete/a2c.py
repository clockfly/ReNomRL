#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import copy, deepcopy
import numpy as np

import renom as rm
from renom import Rmsprop

from renom_rl import AgentBase
from renom_rl.environ.env import BaseEnv
from renom_rl.utility.filter import ProbNodeChooser, MaxNodeChooser
from renom_rl.utility.logger import Logger, A2CLoggerD, AVAILABLE_KEYS

_a2c_keys = AVAILABLE_KEYS["a2c"]["logger"]
_a2c_keys_epoch = AVAILABLE_KEYS["a2c"]["logger_epoch"]


class A2C(AgentBase):
    """A2C class
        This class provides a reinforcement learning agent including training method.
        This class runs on a single thread.

    Args:
        env (BaseEnv): Environment. This must be a child class of ``BaseEnv``.
        network (Model): Actor Critic Model.
        num_worker (int): Number of actor/environment model.
        advantage (int): Advantage steps.
        node_selector (DiscreteNodeChooser): node selector.
        test_node_selector (DiscreteNodeChooser): test node selector.
        loss_func (function): Loss function for train q-network. Default is ``MeanSquaredError()``.
        optimizer: Optimizer for train q-network. Default is ``Rmsprop(lr=0.00025, g=0.95)``.
        entropy_coef(float): Coefficient of actor's output entropy.
        value_coef(float): Coefficient of value loss.
        gamma (float): Discount rate.
        buffer_size (float, int): The size of replay buffer.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> from renom_rl.discrete.a2c import A2C
        >>> from renom_rl.environ.openai import CartPole00
        >>>
        >>> class ActorCritic(rm.Model):
        ...     def __init__(self):
        ...         self.l1=rm.Dense(32)
        ...         self.l2=rm.Dense(32)
        ...         self.l3=rm.Dense(2)
        ...         self.l4=rm.Dense(1)
        ...
        ...     def forward(self,x):
        ...         h1 = self.l1(x)
        ...         h2 = self.l2(h1)
        ...         act = rm.softmax(self.l3(h2))
        ...         val=self.l4(h2)
        ...         return act,val
        ...
        >>> model=ActorCritic()
        >>> env = CartPole00()
        >>> a2c=A2C(env,model)
        >>> a2c.fit(epoch=1,epoch_step=10000)

    References:
        | A. V. Clemente, H. N. Castejon, and A. Chandra.
        | Efficient Parallel Methods for Deep Reinforcement Learning
        | https://arxiv.org/abs/1705.04862



    """

    def __init__(self, env, network, loss_func=None, optimizer=None,
                 gamma=0.99, num_worker=8, advantage=5, value_coef=0.5, entropy_coef=0.01,
                 node_selector=None, test_node_selector=None, gradient_clipping=None, logger=None):
        super(A2C, self).__init__()

        # Reset parameters.
        self._network = network
        self._advantage = advantage
        self._num_worker = num_worker

        self.envs = [deepcopy(env) for i in range(num_worker)]
        self.test_env = env
        self.loss_func = loss_func if loss_func is not None else rm.MeanSquaredError()
        self._optimizer = optimizer if optimizer is not None else rm.Rmsprop(
            0.001, g=0.99, epsilon=1e-10)
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma

        self.node_selector = ProbNodeChooser() if node_selector is None else node_selector
        self.test_node_selector = MaxNodeChooser() if test_node_selector is None else test_node_selector
        self.gradient_clipping = gradient_clipping

        action_shape = env.action_shape
        state_shape = env.state_shape

        self.action_size = action_shape
        self.state_size = state_shape

        # Check Env class type.
        if isinstance(env, BaseEnv):
            action_shape = env.action_shape
            state_shape = env.state_shape
        else:
            raise Exception("Argument env must be a object of BaseEnv class.")

        # Check state and action shape
        assert state_shape == self.envs[0].reset().shape, \
            "Expected state shape is {} but acctual is {}".format(
                state_shape, self.envs[0].reset().shape)

        out_res = self._network(np.zeros((1, *state_shape)))

        assert len(
            out_res) == 2, "( action , value ) are required as returned variables, but the structure is not in that state"

        assert out_res[0].shape[1:] == action_shape, \
            "Expected action shape is {} but acctual is {}".format(
                action_shape, out_res[0].shape[1:])

        assert out_res[1].shape[1:] == (1,), \
            "Expected value shape is {} but acctual is {}".format((1,), out_res[1].shape[1:])

        self._action_shape = action_shape
        self._state_shape = state_shape
        self._initialize()

        # logger
        logger = A2CLoggerD() if logger is None else logger
        assert isinstance(logger, Logger), "Argument logger must be Logger class"
        logger._key_check(log_key=_a2c_keys, log_key_epoch=_a2c_keys_epoch)
        self.logger = logger

    def _initialize(self):
        '''Target network is initialized with same neural network weights of network.'''
        # Reset weight.
        for layer in self._network.iter_models():
            if hasattr(layer, "params"):
                layer.params = {}

    def _action(self, state):
        self._network.set_models(inference=True)
        acts, _ = self._network(state)
        return self.node_selector(acts)

    def _test_action(self, state):
        self._network.set_models(inference=True)
        acts, _ = self._network(state)
        return self.test_node_selector(acts)

    def _value(self, x):
        self._network.set_models(inference=True)
        _, v = self._network(x)
        return v.as_ndarray()

    def _calc_forward(self, x):
        act, val = self._network(x)
        e = - rm.sum(act*rm.log(act+1e-5), axis=1)
        entropy = e.reshape((-1, 1))
        return act, val, entropy

    def fit(self, epoch=1, epoch_step=250000, test_step=None):
        """
        This method executes training of actor critic.
        Test will be runned after each epoch is done.

        Args:
            epoch (int): Number of epoch for training.
            epoch_step (int): Number of step of one epoch.
            test_step (int): Number steps during test.
        """

        # check
        assert isinstance(self.logger, Logger), "logger must be Logger class"
        self.logger._key_check(log_key=_a2c_keys, log_key_epoch=_a2c_keys_epoch)

        # creating local variables
        envs = self.envs
        test_env = self.test_env
        advantage = self._advantage
        threads = self._num_worker
        gamma = self.gamma
        gradient_clipping = self.gradient_clipping
        value_coef = self.value_coef
        entropy_coef = self.entropy_coef

        # env start(after reset)
        [self.envs[_t].start() for _t in range(threads)]

        # logging
        step_counts_log = np.zeros((advantage, threads,))
        step_count = 0
        episode_counts_log = np.zeros((advantage, threads,))
        episode_counts = np.zeros((threads,))

        # epoch
        for e in range(1, epoch + 1):

            # r,a,r,t,s+1
            states = np.zeros((advantage, threads, *test_env.state_shape))
            actions = np.zeros((advantage, threads, 1))
            rewards = np.zeros((advantage, threads, 1))
            dones = np.zeros((advantage, threads, 1))
            states_next = np.zeros((advantage, threads, *test_env.state_shape))

            # value, value next, target value function
            values = np.zeros((advantage, threads, 1))
            target_rewards = np.zeros((advantage, threads, 1))

            # logging
            sum_rewards_log = np.zeros((advantage, threads, ))
            sum_rewards = np.zeros((threads,))
            continuous_steps_log = np.zeros((advantage, threads, ))
            continuous_steps = np.zeros((threads, ))
            epoch_steps_log = np.zeros((advantage, threads, ))
            epoch_steps_j = 0
            nth_episode_counts_log = np.zeros((advantage, threads, ))
            nth_episode_counts = np.zeros((threads, ))

            # env epoch
            _ = [self.envs[_t].epoch() for _t in range(threads)]

            # initiallize
            states[0] = np.array([envs[i].reset() for i in range(threads)]
                                 ).reshape((-1, *test_env.state_shape))

            # action size
            a_, _ = self._network(states[0])
            a_len = len(a_[0].as_ndarray())

            loss = 0

            max_step = epoch_step / advantage

            self.logger._rollout = True
            self.logger.start(epoch_step)

            for j in range(int(max_step)):

                # for each step
                for step in range(advantage):

                    # calculate action value
                    actions[step] = self._action(states[step])

                    # for each thread
                    for thr in range(threads):

                        # next state,reward,done
                        states_next[step][thr], rewards[step][thr], dones[step][thr] = envs[thr].step(
                            int(actions[step][thr]))

                        # summing rewards / append steps
                        sum_rewards[thr] += rewards[step][thr]
                        sum_rewards_log[step][thr] = sum_rewards[thr]
                        continuous_steps[thr] += 1
                        continuous_steps_log[step][thr] = continuous_steps[thr]
                        episode_counts_log[step][thr] = episode_counts[thr]
                        nth_episode_counts_log[step][thr] = nth_episode_counts[thr]

                        # if done, then reset, set next state is initial
                        if dones[step][thr]:
                            states_next[step][thr] = envs[thr].reset()
                            sum_rewards[thr] = 0
                            continuous_steps[thr] = 0
                            episode_counts[thr] += 1
                            nth_episode_counts[thr] += 1

                    epoch_steps_log[step] = epoch_steps_j
                    step_counts_log[step] = step_count

                    # append 1 step
                    epoch_steps_j += 1
                    step_count += 1

                    # setting step to next advanced step
                    if step + 1 < advantage:
                        states[step+1] = states_next[step]

                    # values are calculated at this section
                    values[step] = self._value(states[step])

                    # env epoch step
                    _ = [self.envs[_t].epoch_step() for _t in range(threads)]

                # copy rewards
                target_rewards = np.copy(rewards)

                # calculate rewards
                for i in reversed(range(advantage-1)):
                    target_rewards[i] = rewards[i]+target_rewards[i+1]*gamma*(1-dones[i])

                # -------calcuating gradients-----

                # reshaping states, target
                reshaped_state = states.reshape((-1, *envs[0].state_shape))
                reshaped_target_rewards = target_rewards.reshape((-1, 1))
                advantage_reward = reshaped_target_rewards - values.reshape((-1, 1))

                total_n = advantage * threads

                # reshape index variables for action
                action_index = actions.reshape((-1,))

                # caculate forward with comuptational graph
                self._network.set_models(inference=False)
                with self._network.train():
                    act, val, entropy = self._calc_forward(reshaped_state)
                    act_log = rm.log(act+1e-5)

                    # initiallize
                    action_coefs = np.zeros_like(act.as_ndarray())

                    # write 1 for index at action_coefs
                    action_coefs[range(action_index.shape[0]), action_index.astype("int")] = 1

                    # append act loss and val loss
                    act_loss = rm.sum(- (advantage_reward * action_coefs *
                                         act_log + entropy * entropy_coef) / total_n)

                    val_loss = self.loss_func(val, reshaped_target_rewards) * value_coef

                    # total loss
                    total_loss = val_loss + act_loss

                grad = total_loss.grad()

                if gradient_clipping is not None:
                    gradient_clipping(grad)

                grad.update(self._optimizer)

                val_loss_nd = float(val_loss.as_ndarray())
                entropy_np = float(rm.sum(entropy).as_ndarray())

                singular_list = [epoch_step, e, epoch, val_loss_nd, entropy_np, advantage, threads]
                log1_key = ["max_step", "epoch", "max_epoch",
                            "loss", "entropy", "advantage", "num_worker"]
                log1_value = [[data]*advantage for data in singular_list]

                thread_step_reverse_list = [states, actions, rewards, dones, states_next,
                                            step_counts_log, epoch_steps_log,
                                            episode_counts_log, nth_episode_counts_log,
                                            continuous_steps_log, sum_rewards_log, values]

                log2_key = ["state", "action", "reward", "terminal", "next_state",
                            "total_step", "epoch_step",
                            "total_episode", "epoch_episode",
                            "steps_per_episode", "sum_reward", "values"]

                log2_value = [data.swapaxes(1, 0)[0] for data in thread_step_reverse_list]

                log_dic = {**dict(zip(log1_key, log1_value)), **dict(zip(log2_key, log2_value))}

                self.logger.logger(**log_dic)

                self.logger.update(advantage)

                states[0] = states_next[-1]

                if any([self.envs[_t].terminate() for _t in range(threads)]):
                    print("terminated")
                    break

            else:

                summed_test_reward = self.test(test_step)
                self.logger.logger_epoch(total_episode=episode_counts_log[-1], epoch_episode=nth_episode_counts_log[-1],
                                         epoch=e, max_epoch=epoch, test_reward=summed_test_reward,
                                         advantage=advantage, num_worker=threads)
                self.logger.close()
                continue

            break

    def test(self, test_step=None, **kwargs):
        """
        Test the trained actor agent.

        Args:
            test_step (int, None): Number of steps (not episodes) for test. If None is given, this method tests execute only 1 episode.

        Returns:
            Sum of rewards.
        """

        env = self.test_env

        sum_reward = 0
        env.test_start()
        state = env.reset()

        if test_step is None:
            while True:
                action = self._test_action(state[None, ...])

                state, reward, terminal = env.step(action)

                sum_reward += float(reward)

                env.test_epoch_step()

                if terminal:
                    break

        else:
            for j in range(test_step):

                action = self._test_action(state[None, ...])

                state, reward, terminal = env.step(action)

                sum_reward += float(reward)

                env.test_epoch_step()

                if terminal:
                    env.reset()

        env.test_close()

        return sum_reward
