from __future__ import division
from time import sleep
import copy
from tqdm import tqdm
import numpy as np
import renom as rm
from renom.utility.reinforcement.replaybuffer import ReplayBuffer
from PIL import Image
import numpy as np
import pickle

class DQN(object):
    """DQN class
    This class provides a reinforcement learning agent including training method.

    Args:
        q_network (Model): Q-Network.
        state_size (tuple, list): The size of state.
        action_pattern (int): The number of action pattern.
        gamma (float): Discount rate.
        buffer_size (float, int): The size of replay buffer.
    """

    def __init__(self, env, action_pattern, state_size, q_network = None, loss_func=rm.ClippedMeanSquaredError(),
                 optimizer=rm.Rmsprop(lr=0.00025, g=0.95), gamma=0.99, tau=0.001, batch_size = 32, buffer_size=1e5,
                 greedy_step=1000000, min_greedy=0.0, max_greedy=0.9, test_greedy=0.95):
        
        self._q_network = q_network
        self._target_q_network = copy.deepcopy(self._q_network)
        self._action_size = action_pattern
        self._state_size = state_size
        self._buffer_size = buffer_size
        self._gamma = gamma
        self._buffer = ReplayBuffer([1, ], self._state_size, buffer_size)
        self.env = env 
        self.loss_func = loss_func
        self._optimizer = optimizer
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.greedy_step = greedy_step
        self.min_greedy = min_greedy
        self.max_greedy = max_greedy
        self.test_greedy = test_greedy
        self.greedy = self.min_greedy
        self.g_step = (self.max_greedy - self.min_greedy) / self.greedy_step
        self.initialize()

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
        shape = [-1, ] + list(self._state_size)
        s = state.reshape(shape)
        return np.argmax(self._q_network(s).as_ndarray(), axis=1)
    
    def update(self):
        """This function updates target network."""
        # Check GPU data
        self._target_q_network.copy_params(self._q_network)
        
    def train(self, epoch=50000, batch_size=32, random_step=1000, one_epoch_step=2000, test_step=1000, test_env=None,
              update_period=10000, train_frequency=4, use_pre_train=False):
        """This method executes training of a q-network.
        Training will be done with epsilon-greedy method.

        Args:
            env (function): A function which accepts action as an argument
                and returns prestate, state,  reward and terminal.
            loss_func (Model): Loss function for training q-network.
            optimizer (Optimizer): Optimizer object for training q-network.
            epoch (int): Number of epoch for training.
            batch_size (int): Batch size.
            random_step (int): Number of random step which will be executed before training.
            one_epoch_step (int): Number of step of one epoch.
            test_step (int): Number of test step.
            test_env (function): A environment function for test.
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
            >>> from renom.algorithm.reinforcement.dqn import DQN
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
            ...           epoch=50,
            ...           random_step=5000,
            ...           one_epoch_step=25000,
            ...           test_step=2500,
            ...           test_env=environment,
            ...           optimizer=rm.Rmsprop(lr=0.00025, g=0.95))
            >>>
            Executing random action for 5000 step...
            epoch 000 avg loss:0.0060 avg reward:0.023: 100%|██████████| 25000/25000 [19:12<00:00, 21.70it/s]
                /// Result
                Average train error: 0.006
                Avg train reward in one epoch: 1.488
                Avg test reward in one epoch: 1.216
                Test reward: 63.000
                Greedy: 0.0225
                Buffer: 29537
                ...
            >>>
            >>> print(train_history["train_reward"])

        """
        # History of Learning
        train_reward_list = []
        test_reward_list = []
        train_error_list = []
        
        #env = self.env
        
        '''
        if test_env is None:
            test_env = self.env
        '''
        
        '''
        print("Executing random action for %d step..." % random_step)
        prestate = env.reset()
        for r in range(random_step):
            action = int(np.random.rand() * self._action_size)
            #print(action)
            state, reward, terminal, _ = env.step(action)
            if prestate is not None:
                self._buffer.store(prestate, np.array(action),
                                   np.array(reward), state, np.array(terminal))
        '''
        
        #state = None
        #prestate = None
        count = 0
        #print('started')

        if use_pre_train == True:
            self._q_network.load('my_q_network.h5')
            self._target_q_network.load('my_target_n.h5')
            with open('variables.pkl', 'rb') as f:
                self.greedy, self.g_step = pickle.load(f)
            
        for e in range(epoch):

            initial_state = self.env.reset()
            process_input = self.env.preprocess(initial_state)
            state = np.stack([process_input for _ in range(train_frequency)], axis=0)
            loss = 0
            sum_reward = 0
            train_one_epoch_reward = []
            train_each_epoch_reward = []
            test_one_epoch_reward = []
            test_each_epoch_reward = []
            tq = tqdm(range(one_epoch_step))
            #print('here')
            #print('epoch', e)


            if e%100 == 0.0:
                self._q_network.save('my_q_network.h5')
                self._target_q_network.save('my_target_n.h5')
                with open('variables.pkl', 'wb') as f:
                    pickle.dump([self.greedy, self.g_step], f)
                
            for j in range(one_epoch_step):
                #self.env.render()
                if self.greedy > np.random.rand(): # and state is not None:
                    action = np.argmax(np.atleast_2d(self._q_network(state[None, ...])), axis=1)
                else:
                    action = int(np.random.rand() * self._action_size)
                next_state, reward, terminal, _ = self.env.step(action)
                
                self.greedy += self.g_step
                self.greedy = np.clip(self.greedy, self.min_greedy, self.max_greedy)
                sum_reward += reward
                train_each_epoch_reward.append(reward)
                '''
                if prestate is not None:
                    self._buffer.store(prestate, np.array(action),
                                       np.array(reward), state, np.array(terminal))
                    train_one_epoch_reward.append(reward)
                else:
                    if len(train_one_epoch_reward) > 0:
                        train_each_epoch_reward.append(np.sum(train_one_epoch_reward))
                    train_one_epoch_reward = []

                if j % train_frequency == 0:
                    # Training
                    train_prestate, train_action, train_reward, train_state, train_terminal = \
                        self._buffer.get_minibatch(batch_size)

                    self._q_network.set_models(inference=True)
                    self._target_q_network.set_models(inference=True)

                    target = self._q_network(train_prestate).as_ndarray()
                    target.setflags(write=True)

                    # train_state = train_state.reshape(batch_size, *self._state_size)
                    value = self._target_q_network(train_state).as_ndarray(
                    ) * self._gamma * (~train_terminal[:, None])

                    for i in range(batch_size):
                        a = train_action[i, 0].astype(np.integer)
                        target[i, a] = train_reward[i] + value[i, a]

                    self._network.set_models(inference=False)
                    with self._q_network.train():
                        z = self._q_network(train_prestate)
                        ls = loss_func(z, target)
                    ls.grad().update(optimizer)
                    loss += ls.as_ndarray()

                    if count % update_period == 0:
                        self.update()
                        count = 0
                    count += 1

                msg = "epoch {:03d} loss:{:6.4f} sum reward:{:5.3f}".format(
                    e, float(ls.as_ndarray()), sum_reward)
                tq.set_description(msg)
                tq.update(1)
                '''
                self._buffer.store(state, np.array(action),
                                       np.array(reward), next_state, np.array(terminal))
                train_one_epoch_reward.append(reward)

                if len(self._buffer) > self.batch_size:
                    train_prestate, train_action, train_reward, train_state, train_terminal = \
                        self._buffer.get_minibatch(batch_size)

                    self._q_network.set_models(inference=True)
                    self._target_q_network.set_models(inference=True)

                    target = self._q_network(train_prestate).as_ndarray()
                    target.setflags(write=True)
                    # train_state = train_state.reshape(batch_size, *self._state_size)
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

                    if count % update_period == 0:
                        self.update()
                        count = 0
                        #print('j', j, 'e', e, count, 'updated')
                    count += 1

                #msg = "epoch {:03d} loss:{:6.4f} sum reward:{:5.3f}".format(e, float(ls.as_ndarray()), sum_reward)
                msg = "epoch {:03d} each step reward:{:5.3f}".format(e, reward)
                tq.set_description(msg)
                tq.update(1)

                if terminal == True or j==(one_epoch_step-1):
                    #print('in terminal')
                    train_reward_list.append(sum_reward)
                    train_error_list.append(float(loss) / (j + 1))
                    #print('calling test method')
                    test_total_reward = self.test(test_step, train_frequency)
                    msg = ("epoch {:03d} avg loss:{:6.4f} train_total_reward:{:5.3f} test_total_reward:{:5.3f} e-greedy:{:5.5f}".format(e, float(loss) / (j + 1), sum_reward, test_total_reward, self.greedy))
                    tq.set_description(msg)
                    tq.update(0)
                    tq.refresh()
                    tq.close()
                    tq.write("    /// Result")
                    tq.write("    Average train error: {:5.5f}".format(float(loss) / (j+1)))
                    tq.write("    Avg train reward in one epoch: {:5.5f}".format((float(sum_reward)/(j+1))))
                    tq.write("    Avg test reward in one epoch: {:5.5f}".format(float(test_total_reward)/(j+1)))
                    tq.write("    Total train reward in one epoch: {:5.5f}".format(float(sum_reward)))
                    tq.write("    Test reward: {:5.5f}".format(sum_reward))
                    tq.write("    Greedy: {:5.5f}".format(self.greedy))
                    tq.write("    Buffer: {}".format(len(self._buffer)))
                    sleep(0.25)
                    break

    def test(self, test_step, train_frequency):
        # Test
        #state = None
        #print('in test')
        sum_reward = 0
        initial_state = self.env.reset()
        process_input = self.env.preprocess(initial_state)
        state = np.stack([process_input for _ in range(train_frequency)], axis=0)

        for j in range(test_step):
            if self.test_greedy > np.random.rand():
                action = self.action(state)
            else:
                action = int(np.random.rand() * self._action_size)
            next_state, reward, terminal, _ = self.env.step(action)
            '''
            if prestate is not None:
                test_one_epoch_reward.append(reward)
            else:
                if len(test_one_epoch_reward) > 0:
                    test_each_epoch_reward.append(np.sum(test_one_epoch_reward))
                    test_one_epoch_reward = []
            '''
            sum_reward += float(reward)

            if terminal == True:
                break
            #test_reward_list.append(sum_reward)
            '''
            tq.write("    /// Result")
            tq.write("    Average train error: {:5.3f}".format(float(loss) / one_epoch_step))
            tq.write("    Avg train reward in one epoch: {:5.3f}".format(
                np.mean(train_each_epoch_reward)))
            tq.write("    Avg test reward in one epoch: {:5.3f}".format(
                np.mean(test_each_epoch_reward)))
            tq.write("    Test reward: {:5.3f}".format(sum_reward))
            tq.write("    Greedy: {:1.4f}".format(greedy))
            tq.write("    Buffer: {}".format(len(self._buffer)))

            sleep(0.25)  # This is for jupyter notebook representation.
            '''
        #print('finished test')
        return sum_reward
        ''' 
        return {"train_reward": train_reward_list,
                "train_error": train_error_list,
                "test_reward": test_reward_list}
        '''
