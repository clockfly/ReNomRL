import copy
import numpy as np
from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor
# from threading import BoundedSemaphore

import renom as rm
from renom_rl.environ.env import BaseEnv
from renom import Rmsprop
from renom_rl.utility.gradients import GradientClipping
from renom_rl.utility.filter import ProbNodeChooser, MaxNodeChooser
from renom_rl.utility.event_handler import EventHandler


class A2C(object):
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
        | https://arxiv.org/abs/1509.02971



    """

    def __init__(self, env, network, loss_func=None, optimizer=None,
                 gamma=0.99, num_worker=8, advantage=5, value_coef=0.5, entropy_coef=0.01,
                 node_selector=None, test_node_selector=None, gradient_clipping=None):
        super(A2C, self).__init__()

        self._network = network
        self._advantage = advantage
        self._num_worker = num_worker

        self.envs = [copy.deepcopy(env) for i in range(num_worker)]
        self.test_env = env
        self.loss_func = loss_func if not None else rm.MeanSquaredError()
        self._optimizer = optimizer if not None else rm.Rmsprop(0.001, g=0.99, epsilon=1e-10)
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.events = EventHandler()

        self.node_selector = ProbNodeChooser() if node_selector is None else node_selector
        self.test_node_selector = MaxNodeChooser() if test_node_selector is None else test_node_selector
        self.gradient_clipping = gradient_clipping

        action_shape = env.action_shape
        state_shape = env.state_shape

        self.action_size = action_shape
        self.state_size = state_shape
        self.bar = None

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

    def _initialize(self):
        '''Target q-network is initialized with same neural network weights of q-network.'''
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

        # creating local variables
        envs = self.envs
        test_env = self.test_env
        advantage = self._advantage
        threads = self._num_worker
        gamma = self.gamma
        gradient_clipping = self.gradient_clipping
        value_coef = self.value_coef
        entrpoy_coef = self.entropy_coef

        # env start(after reset)
        [self.envs[_t].start() for _t in range(threads)]

        # logging
        step_count = 0
        episode_counts = np.zeros((threads,))
        sum_rewards = np.zeros((threads,))
        avg_rewards = np.zeros((threads,))

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
            total_rewards = []
            for _ in range(threads):
                total_rewards.append([])
            total_loss_nd = 0
            total_loss_nd_in_epoch = 0
            total_rewards_per_thread = []
            total_rewards_total = 0
            total_rewards_avg = 0

            # initiallize
            states[0] = np.array([envs[i].reset() for i in range(threads)]
                                 ).reshape((-1, *test_env.state_shape))

            # action size
            a_, _ = self._network(states[0])
            a_len = len(a_[0].as_ndarray())

            loss = 0

            max_step = epoch_step / advantage

            tq = tqdm(range(int(max_step)*advantage))

            # env epoch
            [self.envs[_t].epoch() for _t in range(threads)]

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

                        # summing rewards
                        sum_rewards[thr] += rewards[step][thr]

                        # if done, then reset, set next state is initial
                        if dones[step][thr]:
                            states_next[step][thr] = envs[thr].reset()
                            total_rewards[thr].append(sum_rewards[thr])
                            sum_rewards[thr] = 0
                            episode_counts[thr] += 1

                    # append 1 step
                    step_count += advantage

                    # setting step to next advanced step
                    if step + 1 < advantage:
                        states[step+1] = states_next[step]

                    # values are calculated at this section
                    values[step] = self._value(states[step])

                # env epoch step
                [self.envs[_t].epoch_step() for _t in range(threads)]

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
                    val_loss = self.loss_func(reshaped_target_rewards, val)

                    # total loss
                    total_loss = val_loss + act_loss

                # calc
                total_rewards_per_thread = [np.sum(x) for x in total_rewards]

                grad = total_loss.grad()

                if gradient_clipping is not None:
                    gradient_clipping(grad)

                grad.update(self._optimizer)

                total_loss_nd = float(total_loss.as_ndarray())
                total_loss_nd_in_epoch += total_loss_nd

                # set next_state
                for thr in range(threads):
                    states[0][thr] = states_next[-1][thr]

                # message print
                msg = "agent{}, epoch {:04d} loss {:5.4f} rewards in epoch {:4.3f}  episode {:04.1f} rewards in episode {:4.3f}."\
                    .format(0, e, abs(total_loss_nd), total_rewards_per_thread[0], episode_counts[0],
                            total_rewards[0][-1] if len(total_rewards[0]) > 0 else 0)

                # description
                tq.set_description(msg)
                tq.update(advantage)

                # event handler
                self.events.on("step", e, j, step)

                if any([self.envs[_t].terminate() for _t in range(threads)]):
                    print("terminate")
                    break

            else:

                summed_test_reward = self.test(test_step)

                total_rewards_total = np.sum(total_rewards_per_thread)
                total_rewards_avg = total_rewards_total / np.sum(episode_counts)

                msg = "epoch {:03d} avg_loss:{:6.4f} total reward in epoch: [train:{:4.3f} test:{:4.3}] " + \
                    "avg train reward in episode:{:4.3f}"
                msg = msg.format(e, total_loss_nd_in_epoch/max_step, total_rewards_total,
                                 summed_test_reward, total_rewards_avg)

                self.events.on("end_epoch", e, j, self, total_loss_nd_in_epoch/max_step, total_rewards_total,
                               summed_test_reward)

                tq.set_description(msg)
                tq.update(0)
                tq.refresh()
                tq.close()
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
