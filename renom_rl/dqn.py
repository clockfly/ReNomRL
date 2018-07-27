from __future__ import division
from time import sleep
import copy
from tqdm import tqdm
import numpy as np
import renom as rm
from renom.utility.reinforcement.replaybuffer import ReplayBuffer
from renom_rl.env import BaseEnv
from PIL import Image
import numpy as np
import pickle

try:
    from gym.core import Env as OpenAIEnv
except:
    OpenAIEnv = None


class DQN(object):
    """DQN class
    This class provides a reinforcement learning agent including training method.

    Args:
        env (Environment):
        q_network (Model): Q-Network.
        action_pattern (int): The number of action pattern.
        state_size (tuple, list): The size of state.
        loss_func (function):
        optimizer: 
        gamma (float): Discount rate.
        tau (float):
        buffer_size (float, int): The size of replay buffer.
    """

    def __init__(self, env, q_network, loss_func=rm.ClippedMeanSquaredError(),
                 optimizer=rm.Rmsprop(lr=0.00025, g=0.95), gamma=0.99, tau=0.0, buffer_size=1e6):

        self._q_network = q_network
        self._target_q_network = copy.deepcopy(self._q_network)
        self._buffer_size = buffer_size
        self._gamma = gamma
        self.env = env
        self.loss_func = loss_func
        self._optimizer = optimizer
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.tau = tau
        self.initialize()

        # Train histories
        self.train_reward_list = []
        self.test_reward_list = []

        if isinstance(env, BaseEnv):
            action_shape = env.action_shape
            state_shape = env.state_shape
        elif isinstance(env, OpenAIEnv):
            action_shape = env.action_space.n
            state_shape = env.observation_space.shape
        else:
            raise Exception("Argument env must be a object of BaseEnv class or OpenAI gym Env.")

        # Check env object
        # Check sample method.
        sample = self.env.sample()
        assert isinstance(sample, int), \
            "Sampled action from env object must be scalar and integer type. Actual is {}".format(
                type(sample))

        # Check state and action shape
        assert state_shape == self.env.reset().shape
        action = self._q_network(np.zeros((1, *state_shape))).as_ndarray()
        assert action.shape[1] == action_shape
        #####

        self._action_size = action_shape
        self._state_size = state_shape
        self._buffer = ReplayBuffer([1, ], self._state_size, buffer_size)

    def initialize(self):
        '''target q-network is initialized with same neural network weights as q-network'''
        self._target_q_network.copy_params(self._q_network)

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

    def update(self):
        """This function updates target network."""
        if self.tau == 0:
            self._target_q_network.copy_params(self._q_network)
        else:
            for ql, tql in zip(self._walk_model(self._q_network),
                               self._walk_model(self._target_q_network)):
                if not hasattr(ql, 'params'):
                    continue
                for k in ql.params.keys():
                    tql.params[k] = ql.params[k] * self.tau + tql.params[k] * (1 - self.tau)


    def fit(self, episode=50000, batch_size=64, episode_step=2000, random_step=5000, test_step=1000, update_period=10000, train_frequency=4, min_greedy=0.0, max_greedy=0.9, greedy_step=1000000, test_greedy=0.95, test_period=50, render=False):
        """This method executes training of a q-network.
        Training will be done with epsilon-greedy method.

        Args:
            env (function): A function which accepts action as an argument
                and returns prestate, state,  reward and terminal.
            loss_func (Model): Loss function for training q-network.
            optimizer (Optimizer): Optimizer object for training q-network.
            episode (int): Number of episode for training.
            batch_size (int): Batch size.
            random_step (int): Number of random step which will be executed before training.
            episode_step (int): Number of step of one episode.
            test_step (int): Number of test step.
            update_period (int): Period of updating target network.
            greedy_step (int): Number of step
            min_greedy (int): Minimum greedy value
            max_greedy (int): Maximum greedy value
            test_greedy (int): Greedy threshold
            train_frequency (int): For the learning step, training is done at this cycle

        Returns:
            (dict): A dictionary which includes reward list of training and loss list.

        Example:
            >>> import renom as rm
            >>> from renom_rl.dqn import DQN
            >>>
            >>> q_network = rm.Sequential([
            ...    rm.Conv2d(32, filter=8, stride=4),
            ...    rm.Relu(),
            ...    rm.Conv2d(64, filter=4, stride=2),
            ...    rm.Relu(),
            ...    rm.Conv2d(64, filter=3, stride=1),
            ...    rm.Relu(),
            ...    rm.Flatten(),
            ...    rm.Dense(512),
            ...    rm.Relu(),
            ...    rm.Dense(action_pattern)
            ... ])
            >>>
            >>> state_size = (4, 84, 84)
            >>> action_pattern = 4
            >>>
            >>> def environment(action):
            ...     prestate = ...
            ...     state = ...
            ...     reward = ...
            ...     terminal = ...
            ...     return prestate, state, reward, terminal
            >>>
            >>> # Instantiation of DQN object
            >>> dqn = DQN(model,
            ...           state_size=state_size,
            ...           action_pattern=action_pattern,
            ...           gamma=0.99,
            ...           buffer_size=buffer_size)
            >>>
            >>> # Training
            >>> train_history = dqn.train(environment,
            ...           loss_func=rm.ClippedMeanSquaredError(clip=(-1, 1)),
            ...           episode=50,
            ...           random_step=5000,
            ...           episode_step=25000,
            ...           test_step=2500,
            ...           optimizer=rm.Rmsprop(lr=0.00025, g=0.95))
            >>>
            Executing random action for 5000 step...
            episode 000 avg loss:0.0060 avg reward:0.023: 100%|██████████| 25000/25000 [19:12<00:00, 21.70it/s]
                /// Result
                Average train error: 0.006
                Avg train reward in one episode: 1.488
                Avg test reward in one episode: 1.216
                Test reward: 63.000
                Greedy: 0.0225
                Buffer: 29537
                ...
            >>>
            >>> print(train_history["train_reward"])

        """
        greedy = min_greedy
        g_step = (max_greedy - min_greedy) / greedy_step

        print("Run random {} step for storing experiences".format(random_step))

        state = self.env.reset()
        for i in range(1, random_step + 1):
            action = self.env.sample()
            if isinstance(self.env, BaseEnv):
                next_state, reward, terminal = self.env.step(action)
            elif isinstance(self.env, OpenAIEnv):
                next_state, reward, terminal, _ = self.env.step(action)
            else:
                raise Exception("Argument env must be a object of BaseEnv class or OpenAI gym Env.")

            self._buffer.store(state, np.array(action),
                               np.array(reward), next_state, np.array(terminal))
            if terminal:
                state = self.env.reset()

        # History of Learning
        train_reward_list = []
        test_reward_list = []
        train_error_list = []

        count = 0
        for e in range(1, episode + 1):
            loss = 0
            sum_reward = 0
            state = self.env.reset()
            train_one_episode_reward = []
            train_each_episode_reward = []
            test_one_episode_reward = []
            test_each_episode_reward = []
            tq = tqdm()

            for j in range(episode_step):
                if greedy > np.random.rand():  # and state is not None:
                    action = np.argmax(np.atleast_2d(self._q_network(
                        state[None, ...]).as_ndarray()), axis=1)
                else:
                    action = self.env.sample()

                if isinstance(self.env, BaseEnv):
                    next_state, reward, terminal = self.env.step(action)
                else:
                    next_state, reward, terminal, _ = self.env.step(action)

                greedy += g_step
                greedy = np.clip(greedy, min_greedy, max_greedy)
                sum_reward += reward
                train_each_episode_reward.append(reward)
                self._buffer.store(state, np.array(action),
                                   np.array(reward), next_state, np.array(terminal))
                train_one_episode_reward.append(reward)
                state = next_state

                if len(self._buffer) > batch_size:
                    train_prestate, train_action, train_reward, train_state, train_terminal = \
                        self._buffer.get_minibatch(batch_size)

                    self._q_network.set_models(inference=True)
                    self._target_q_network.set_models(inference=True)

                    target = self._q_network(train_prestate).as_ndarray()
                    target.setflags(write=True)
                    value = self._target_q_network(train_state).as_ndarray(
                    ) * self._gamma * (~train_terminal[:, None])

                    for i in range(batch_size):
                        a = train_action[i, 0].astype(np.integer)
                        target[i, a] = train_reward[i] + value[i, a]

                    self._q_network.set_models(inference=False)
                    with self._q_network.train():
                        z = self._q_network(train_prestate)
                        ls = self.loss_func(z, target)
                    ls.grad().update(self._optimizer)
                    loss += ls.as_ndarray()

                    if count % update_period == 0 or self.tau > 0:
                        self.update()
                        count = 0
                    count += 1

                msg = "episode {:03d} each step reward:{:5.3f}".format(e, sum_reward)
                tq.set_description(msg)
                tq.update(1)
                if terminal:
                    break

            state = self.env.reset()
            train_reward_list.append(sum_reward)
            self.train_reward_list.append(sum_reward)
            train_error_list.append(float(loss) / (j + 1))
            msg = ("episode {:03d} avg_loss:{:6.3f} total_reward [train:{:5.3f} test:-] e-greedy:{:5.3f}".format(
                e, float(loss) / (j + 1), sum_reward, greedy))
            if e % test_period == 0 and e:
                test_total_reward = self.test(test_step, test_greedy, render)
                self.test_reward_list.append(test_total_reward)
                msg = ("episode {:03d} avg_loss:{:6.3f} total_reward [train:{:5.3f} test:{:5.3f}] e-greedy:{:5.3f}".format(
                    e, float(loss) / (j + 1), sum_reward, test_total_reward, greedy))
            tq.set_description(msg)
            tq.update(0)
            tq.refresh()
            tq.close()
            sleep(0.05)

    def test(self, test_step=2000, test_greedy=0.95, render=False):
        # Test
        sum_reward = 0
        state = self.env.reset()

        for j in range(test_step):
            if test_greedy > np.random.rand():
                action = self.action(state)
            else:
                action = self.env.sample()
            if isinstance(self.env, BaseEnv):
                state, reward, terminal = self.env.step(action)
            else:
                state, reward, terminal, _ = self.env.step(action)
            sum_reward += float(reward)
            if render:
                self.env.render()
            if terminal:
                break
        return sum_reward

