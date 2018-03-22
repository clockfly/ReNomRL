from time import sleep
import copy
from tqdm import tqdm
import numpy as np
import renom as rm
from renom.utility.reinforcement.replaybuffer import ReplayBuffer
from renom.utility.initializer import Uniform, GlorotUniform
import time

class OU_noise(object):
    """ 
    DDPG paper ornstein-uhlenbeck noise parameters are theta=0.15, sigma=0.2 
    """
    
    def __init__(self, env, theta = 0.15, sigma = 0.2, x0 = None):
        # x0 is action_space size
        self.mu = np.zeros(shape=(1,env.action_space.shape[0]))
        self.theta = theta
        self.sigma = sigma
        self.dt = 1e-2
        self.x0 = np.zeros(shape=(1,self.mu.shape[0])) if x0 is None else x0
        
    def sample(self):
        self.x0 = self.x0 + self.theta*(self.mu - self.x0)*self.dt + self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)  
        return self.x0

class Actor(rm.Model):
    '''Actor network Default
    '''
    def __init__(self, env):
        self._layers = []
        self.env = env
        self._l1 = rm.Dense(400, initializer=GlorotUniform())
        self._l2 = rm.Dense(300, initializer=GlorotUniform())
        self._l3 = rm.Dense(self.env.action_space.shape[0], initializer=Uniform(min=-0.003, max=0.003))
        self._layers = [self._l1, self._l2, self._l3]
    
      
    def forward(self, x):
        ''' forward propagation to get action '''
        h1 = rm.relu(self._l1(x))
        h2 = rm.relu(self._l2(h1))
        h3 = rm.tanh(self._l3(h2))
        #h3 = self._l3(h2)
        h = h3*self.env.action_space.high[0]
        
        return h
    
    
    def weigiht_decay(self):
        '''For L2-norm to minimize the overfitting (it is an optional)'''
        weight_decay = 0
        for i in range(len(self._layers)):
            weight_decay += rm.sum(self._layers[i].params.w**2)
        return weight_decay


class Critic(rm.Model):
    '''Critic network Default'''
    def __init__(self, env):        
        self._layers = []
        self.env = env
        self._l1 = rm.Dense(400, initializer=GlorotUniform())
        self._l2 = rm.Dense(300, initializer=GlorotUniform())
        self._l3 = rm.Dense(1, initializer=Uniform(min=-0.003, max=0.003))
        self._layers = [self._l1, self._l2, self._l3]
    
        
    def forward(self, x, action):
        '''Forward propagation'''
        h1 = rm.relu(self._l1(x))
        h2 = rm.relu(self._l2(rm.concat(h1,action))) # actions are applied at 2nd hidden layer
        h = self._l3(h2)        
        return h
    
    
    def weigiht_decay(self):
        '''To minimize over fitting L2-norm (it is an optional)'''
        weight_decay = 0
        for i in range(len(self._layers)):
            weight_decay += rm.sum(self._layers[i].params.w**2)
        return weight_decay

class DDPG(object):
    """
    DDPG class
    This class provides a reinforcement learning agent including training and testing methods.
    Args:
        env (openAI Gym): An instance of Environment to be learned
                        for Example, Pendulum-v0 environment, has methods reset, step.
                        env.reset() --> resets initial state of environment
                        env.step(action) --> take action value and returns (next_state, reward, terminal, _)
        actor_network (Model): Actor-Network. If it is None, default ANN is created 
                                with [400, 300] hidden layer sizes
        critic_network (Model): basically a Q(s,a) function Network.
        actor_optimizer : Adam
        critic_optimizer : Adam
        gamma (float): Discount rate.
        tau (float): target_networks update parameter
        batch_size (int): mini batch size
        buffer_size (float, int): The size of replay buffer.
        initliaze(): To make the tareget network's actor & critic weights same as actor & critic network weights
    """

    def __init__(self, env, actor_network = None, critic_network = None, loss_func=rm.mse, \
                 actor_optimizer=rm.Adam(0.0001),critic_optimizer=rm.Adam(0.001), gamma=0.99, \
                 tau=0.001, batch_size= 64, buffer_size=100000, l2_decay = 0.0):
        
        if actor_network == None:
            self._actor = Actor(env=env)
            self._target_actor = copy.deepcopy(self._actor)
        else:
            self._actor = actor_network
            self._target_actor = copy.deepcopy(self._critic)
        
        if critic_network == None:
            self._critic = Critic(env=env)
            self._target_critic = Critic(env=env)
        else:
            self._critic = critic_network
            self._target_critic = target_critic_network
        
        self.loss_func = loss_func
        self._actor_optimizer = sgd(lr = actor_lr )
        self._critic_optimizer = sgd(lr = critic_lr)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self._state_size = env.observation_space.shape
        self._buffer = ReplayBuffer([1,], self._state_size, buffer_size)
        
        self.gamma = gamma
        self.env = env
        self.tau = tau
        self.l2_decay = l2_decay
        self.initalize()
        
    def action(self, state):
        """This method returns an action according to the given state.
        Args:
            state (ndarray): A state of an environment.

        Returns:
            (int, ndarray): Action.
        """
        self._actor.set_models(inference=True)
        shape = [-1, ] + list(self._state_size)
        s = state.reshape(shape)
        return self._actor(s).as_ndarray()
        
    def train(self, num_episodes=1000, num_steps=200, render=False):
        """ This method executes training of an actor-network.
        Here, target actor & critic network weights are updated after every actor & critic update using self.tau
        Args:
            num_episodes (int): training number of epsiodes
            num_steps (int): Depends on the type of Environment in-built setting.
                             Environment reaches terminal situatuion in two cases.
                            (i) In the type of Game, it is game over
                            (ii) Maximum time steps to play                    
        Returns:
            (dict): A dictionary which includes reward list of training and loss list.
        """
        x0 = np.zeros(shape=(1,self.env.action_space.shape[0]))
        noise = OU_noise(env=self.env, x0=x0) # DDPG specific noise
        
        reward_list = []
        critic_loss_list = []
        
        for i in range(num_episodes):
            s = self.env.reset()
            ep_reward = 0.0
            cumulative_critic_loss = 0.0
            loss = 0.0
            tq = tqdm(range(num_steps))
            for j in range(num_steps):
                if render:
                    self.env.render()
                prestate = np.reshape(s,(1, self.env.observation_space.shape[0]))
                
                action = self._actor.forward(prestate) + noise.sample()
                
                state, reward, terminal, _ = self.env.step(action[0])
                
                self._buffer.store(prestate, np.array(action),
                                       np.array(reward), state, np.array(terminal))
                
                if len(self._buffer) > self.batch_size:
                    
                    train_prestate, train_action, train_reward, train_state, train_terminal = \
                        self._buffer.get_minibatch(self.batch_size)
                    
                    qmax = self.target_value_function(train_state).as_ndarray()
                    
                    target_q = []
                    for k in range(self.batch_size):
                        if train_terminal[k]:
                            target_q.append(np.asarray([train_reward[k]]))
                        else:
                            rr = train_reward[k] + (self.gamma*qmax[k])
                            target_q.append(rr)
                            
                    target_q = np.asarray(target_q).reshape(self.batch_size,1)
                    
                    with self._critic.train():
                        value = self._critic.forward(train_prestate, train_action)
                        critic_loss = self.loss_func(value, target_q) + self.l2_decay*self._critic.weigiht_decay()
                    loss += critic_loss.as_ndarray()
                    critic_loss.grad().update(self._critic_optimizer)
                    cumulative_critic_loss += critic_loss.as_ndarray()
                    
                    with self._actor.train():
                        actor_loss = self.value_function(train_prestate) + self.l2_decay*self._actor.weigiht_decay()
                    
                    with self._critic.prevent_update():
                        actor_loss.grad(-1*np.ones_like(actor_loss)).update(self._actor_optimizer)
                    
                    self.update()

                s = state
                ep_reward += reward
                tq.set_description("episode: {:03d} Each step reward:{:0.2f}".format(i, reward))
                tq.update(1)
                if terminal:
                    #if i%self.every == 0.0:
                    #    print('episode',i, 'episode total reward ', ep_reward)
                    reward_list.append(ep_reward)
                    critic_loss_list.append(cumulative_critic_loss)
                    tq.set_description("episode: {:03d} Total reward:{:0.2f} avg loss:{:6.4f}".format(i, ep_reward, float(loss) / (j + 1)))
                    tq.update(0)
                    tq.refresh()
                    tq.close()
                    break
        
        return (reward_list, critic_loss)
    
    
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
        self._target_actor.copy_params(self._actor)
        self._target_critic.copy_params(self._critic)
        
    def update(self):
        '''updare target networks'''
        for al, tal in zip(self._actor._layers, self._target_actor._layers):
            for k in al.params.keys():
                tal.params[k] = al.params[k]*self.tau + tal.params[k]*(1 - self.tau)
                
        for cl, tcl in zip(self._critic._layers, self._target_critic._layers):
            for k in cl.params.keys():
                tcl.params[k] = cl.params[k]*self.tau + tcl.params[k]*(1 - self.tau)
    
    
    def test(self, episodes=5, num_steps=200 ,render=False):
        '''test the trained network
        Args:
            epsiodes: number of trail episodes to run
            render: If it is true you can see how the Environment position at each time.
        Return:
            (list): A list of cumulative test rewards
        '''
        ep_test_reward_list = []
        
        for i in range(episodes):
            ep_reward = 0
            prestate = self.env.reset() 
            #print('initial state', current_state)
            if render:
                self.env.render()
                #time.sleep(5) # to observe the initial state
            tq = tqdm(range(num_steps))
            for _ in range(num_steps):
                if render:
                    self.env.render()
                a = self._actor.forward(np.reshape(prestate, (1, self.env.observation_space.shape[0])))
                state, reward, terminal, _ = self.env.step(a[0])
                ep_reward += reward
                prestate = state
                tq.set_description("episode: {:03d} Each step reward:{:0.2f}".format(i, ep_reward))
                tq.update(1)
                if terminal:
                    ep_test_reward_list.append(ep_reward)
                    tq.set_description("episode: {:03d} Total reward:{:0.2f}".format(i, ep_reward))
                    tq.update(0)
                    tq.refresh()
                    tq.close()
                    break
                
        return ep_test_reward_list
