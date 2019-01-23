from copy import copy, deepcopy
import numpy as np


class RollOut(object):
    """
    RollOut memory allows developers to save, reset, calculate etcs from multiple agents.
    must input tuple obect for state memory shape when creating an instance.

    :default memory:

        - states: advantage*threads*shape
        - actions: advantage*threads*1
        - rewards: advantage*threads*1
        - dones: advantage*threads*1
        - states_next: advantage*threads*shape

    ..code-block:: python

        # 5 advantage steps, 16 threads, 4,84,84 state space
        rollout=RollOut(5,16,(4,84,84))

    """

    def __init__(self, advantage, threads, shape):

        # for reset
        self.advantage = advantage
        self.threads = threads
        self.shape = shape

        # r,a,r,t,s+1
        self.states = np.zeros((advantage, threads, *shape))
        self.actions = np.zeros((advantage, threads, 1))
        self.rewards = np.zeros((advantage, threads, 1))
        self.dones = np.zeros((advantage, threads, 1))
        self.states_next = np.zeros((advantage, threads, *shape))

        # for step_refresh
        self.i = 0
        self.max_step = advantage

    def reset(self):
        """
        This function resets all the saved memories to zeros.
        """

        # for reset
        advantage = self.advantage
        threads = self.threads
        shape = self.shape

        # r,a,r,t,s+1
        self.states = np.zeros((advantage, threads, *shape))
        self.actions = np.zeros((advantage, threads, 1))
        self.rewards = np.zeros((advantage, threads, 1))
        self.dones = np.zeros((advantage, threads, 1))
        self.states_next = np.zeros((advantage, threads, *shape))

    def insert(self, step, thr, state, reward, done):
        """
        This function allows state, reward, and done variables into memory self.

        Args:

            step: advanced step number
            thr: thread
            state: numpy variables
            reward: the reward obtained from that state
            done: flag that represent whether env is done or not

        """
        # inserting values for states rewards dones
        assert state.shape == self.shape, "insert state is not a numpy or shape is wrong"
        assert isinstance(float(reward), float), "insert reward is not a numpy or shape is not 1"
        assert isinstance(float(done), float), "insert done is not a numpy or shape is not 1"

        self.states_next[step][thr] = state
        self.rewards[step][thr] = reward
        self.dones[step][thr] = done

    def step_refresh(self):
        """
        This function copies the states_next memory to states memory and increments one step.
        """
        # inserting states_next to states

        i = self.i

        if i + 1 < self.max_step:
            self.states[i+1] = self.states_next[i]
        else:
            self.states[0] = self.states_next[i]

        self.i = (i+1) % self.max_step

    def get_1darray(self, key):
        """
        This function allows developers to get values from memory.
        key can be list of strings or a string. this is for geting 1d array list.
        """
        res_b = self.get(key)

        if isinstance(key, list):
            res = []
            for r in res_b:
                res.append(r.reshape(-1,))
            res = tuple(res)

        else:
            res = res_b

        return res


class RollOutA2C(RollOut):
    """
    This class is specified for A2C.
    target rewards and values are appended to the memory dicitonary.
    """

    def __init__(self, advantage, threads, shape, gamma):

        super(RollOutA3C, self).__init__(advantage, threads, shape)

        self.values = np.zeros((advantage, threads, 1))
        self.target_rewards = np.zeros((advantage, threads, 1))

        self.gamma = gamma

    def calc_target(self):
        """
        This calculates target_rewards.
        """

        advantage = self.advantage
        dones = self.dones
        rewards = self.rewards
        gamma = self.gamma

        # copy rewards
        self.mem["target_rewards"] = copy(rewards)

        # calculate rewards
        for i in reversed(range(advantage-1)):
            self.mem["target_rewards"][i] = rewards[i] + \
                self.mem["target_rewards"][i+1]*gamma*(1-dones[i])
