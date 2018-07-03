
import copy
import numpy as np
from tqdm import tqdm
from time import sleep

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


class DDPG(object):
    """DDPG class
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

    def __init__(self, env, actor_network, critic_network, loss_func=rm.mse,
                 actor_optimizer=rm.Adam(0.0001), critic_optimizer=rm.Adam(0.001), gamma=0.99,
                 tau=0.001, buffer_size=1e6):

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
        self.initalize()

        if isinstance(env, BaseEnv):
            action_shape = env.action_shape
            state_shape = env.state_shape
            if isinstance(action_shape, 'int'):
                action_shape = tuple(action_shape)
            if isinstance(state_shape, 'int'):
                state_shape = tuple(state_shape)
        elif isinstance(env, OpenAIEnv):
            action_shape = env.action_space.shape
            state_shape = env.observation_space.shape
        else:
            raise Exception("Argument env must be a object of BaseEnv class or OpenAI gym Env.")

        # Check env object
        # Check sample method.
        if isinstance(env, BaseEnv):
            sample = self.env.sample()
        elif isinstance(env, OpenAIEnv):
            sample = self.env.action_space.sample()
        else:
            raise Exception("Argument env must be a object of BaseEnv class or OpenAI gym Env.")

        assert isinstance(sample, np.ndarray), \
            "Sampled action from env object must be numpy ndarray. Actual is {}".format(
                type(sample))

        # Check state and action shape
        assert state_shape == self.env.reset().shape
        action_sample = self._actor(np.zeros((1, *state_shape))).as_ndarray()
        assert action_sample.shape[1:] == action_shape
        #####

        self.action_size = action_shape
        self.state_size = state_shape
        self._buffer = ReplayBuffer(self.action_size, self.state_size, buffer_size)

    def action(self, state):
        """This method returns an action according to the given state.
        Args:
            state (ndarray): A state of an environment.

        Returns:
            (int, ndarray): Action.
        """
        self._actor.set_models(inference=True)
        return self._actor(state.reshape(1, *self.state_size)).as_ndarray()

    def train(self, episode=1000, episode_step=2000, batch_size=32, random_step=1000,
              test_step=2000, train_frequency=1, min_exploration_rate=0.01, max_exploration_rate=1.0,
              exploration_step=10000, test_period=10, noise=OU()):
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

        e_rate = max_exploration_rate
        e_step = (min_exploration_rate - max_exploration_rate) / exploration_step

        state = self.env.reset()
        for _ in range(1, random_step + 1):
            if isinstance(self.env, BaseEnv):
                action = self.env.sample()
                next_state, reward, terminal = self.env.step(action)
            elif isinstance(self.env, OpenAIEnv):
                action = self.env.action_space.sample()
                next_state, reward, terminal, _ = self.env.step(action)
            else:
                raise Exception("Argument env must be a object of BaseEnv class or OpenAI gym Env.")
            self._buffer.store(state, action,
                               np.array(reward), next_state, np.array(terminal))
            if terminal:
                state = self.env.reset()
            else:
                state = next_state

        train_reward_list = []

        for e in range(1, episode + 1):
            state = self.env.reset()
            sum_reward = 0.0
            loss = 0.0
            tq = tqdm(range(episode_step))
            for j in range(episode_step):
                action = self.action(state)
                sampled_noise = noise.sample(action) * e_rate
                action += sampled_noise

                if isinstance(self.env, BaseEnv):
                    next_state, reward, terminal = self.env.step(action)
                elif isinstance(self.env, OpenAIEnv):
                    next_state, reward, terminal, _ = self.env.step(action)

                self._buffer.store(state, action,
                                   np.array(reward), next_state, np.array(terminal))
                sum_reward += reward
                state = next_state
                e_rate += e_step
                e_rate = np.clip(e_rate, min_exploration_rate, max_exploration_rate)

                if len(self._buffer) > batch_size and j%train_frequency == 0:
                    train_prestate, train_action, train_reward, train_state, train_terminal = \
                        self._buffer.get_minibatch(batch_size)

                    qmax = self.target_value_function(train_state).as_ndarray()

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
                        value = self._critic.forward(train_prestate, train_action)
                        critic_loss = self.loss_func(value, target_q)
                    critic_loss.grad().update(self._critic_optimizer)
                    loss += critic_loss.as_ndarray()

                    self._actor.set_models(inference=False)
                    with self._actor.train():
                        actor_loss = self.value_function(train_prestate)
                    actor_loss.grad(-1 * np.ones_like(actor_loss)).update(self._actor_optimizer)
                    self.update()

                sum_reward = float(sum_reward)
                tq.set_description("episode: {:03d} Each step reward:{:0.2f}".format(e, sum_reward))
                tq.update(1)

                if terminal:
                    break

            train_reward_list.append(sum_reward)
            msg = "episode {:03d} avg_loss:{:6.3f} total_reward [train:{:5.3f} test:-] exploration:{:5.3f}"
            tq.set_description(msg.format(e, float(loss) / (j + 1), sum_reward, e_rate))
            if e % test_period == 0:
                tq.set_description("Running test for {} step".format(test_step))
                tq.update(0)
                test_total_reward = self.test(test_step)
                msg = "episode {:03d} avg_loss:{:6.3f} total_reward [train:{:5.3f} test:{:5.3f}] exploration:{:5.3f}"
                tq.set_description(msg.format(e, float(loss) / (j + 1),
                                              sum_reward, test_total_reward, e_rate))
            tq.update(0)
            tq.refresh()
            tq.close()

    def value_function(self, state):
        '''Value of predict network Q_predict(s,a)
        Args:
            state: input state
        Returns:
            value: Q(s,a) value
        '''
        action = self._actor.forward(state)
        value = self._critic.forward(state, action)
        return value

    def target_value_function(self, state):
        '''Value of target network Q_target(s,a).
        Args:
            state: input state
        Returns:
            value: Q(s,a) value
        '''
        action = self._target_actor.forward(state)
        value = self._target_critic.forward(state, action)
        return value

    def initalize(self):
        '''target actor and critic networks are initialized with same neural network weights as actor & critic network'''
        for layer in list(self._walk_model(self._target_critic)) + list(self._walk_model(self._target_actor)):
            if hasattr(layer, "params"):
                layer.params = {}
        self._target_actor.copy_params(self._actor)
        self._target_critic.copy_params(self._critic)

    def _walk_model(self, model):
        yield self
        for k, v in sorted(self.__dict__.items(), key=lambda x: x[0]):
            if isinstance(v, rm.Model):
                yield self._walk_model(v)

    def update(self):
        '''updare target networks'''
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

    def test(self, test_steps=2000):
        '''test the trained network
        Args:
        Return:
            (list): A list of cumulative test rewards
        '''
        sum_reward = 0
        state = self.env.reset()
        for _ in range(test_steps):
            action = self._actor.forward(state.reshape(1, *self.state_size)).as_ndarray()[0]
            if isinstance(self.env, BaseEnv):
                state, reward, terminal = self.env.step(action)
            else:
                state, reward, terminal, _ = self.env.step(action)
            sum_reward += reward
            if terminal:
                break
        return sum_reward
