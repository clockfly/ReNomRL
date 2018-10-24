import numpy as np
from renom_rl.function.epsilon import EpsilonUpdate
from renom_rl.function.noise import OU,GP

def check_kwargs_exist(kwargs=None,*key):
    assert all(k in kwargs for k in key), "you have set {} for this step_mode".format(key)

def get_keys(kwargs=None,*keys):
    res= {}
    for k in keys:
      if k in kwargs:
        res[i]=kwargs[i]
    return res



class ActionFilter(object):
    """Action Filter
    This is the class of action filter. Action Filter allows the Agent to explore instead of being deterministic.
    Class such as epsilon greedy have this as its parent object. Users will get an error if this was not used as the filter.
    `__call__``test``value` are required.
    """

    def __call__(self, action_greedy, action_random,
                 step=None, eposode=None, epoch=None):
        raise NotImplemented

    def test(self, greedy_action, random_action):
        raise NotImplemented

    def value(self):
        raise NotImplemented


class EpsilonGreedyFilter(ActionFilter):
    """Epsilon Greedy Filter
    This class allow you to change the epsilon function. These are implemented in RL framework.
    By specifying the step_mode ("step_linear","episode_base" etc.), you can change the epsilon mode.
    All functions are in the range of min and max. This function uses the `EpsilonUpdate` class.
    Check the `EpsilonUpdate` class for more details.

    Args:
        **kwargs: parameters such as `min`, `max`, `epsilon_step`, `episode`, `alpha`, `test_greedy`

    Example:
        >>> epsilon = EpsilonUpdate(mode="step_linear",initial=0,min=0,max=0.9,epsilon_step=250000)
        >>> print(epsilon.value()) # this will print the initial epsilon which is 0
    """

    def __init__(self, **kwargs):

        self.func=EpsilonUpdate(**kwargs)
        self.test_greedy=kwargs["test_greedy"] if "test_greedy" in kwargs else 0


    def __call__(self, greedy_action, random_action,
                 step=None, episode=None, epoch=None):
        """
        Args:
            greedy_action(float): action that is outputted from the agent.
            random_action(float): random action variable. Variable used from BaseEnv.sample.
            step(int): total accumulated steps (irrelevent to episodes)
            episode(int): total accumulated episodes (irrelevent to epoch)
            epoch(int): epoch steps
        """

        greedy_ratio =self.func(step,episode,epoch)

        if np.random.rand() > greedy_ratio:
            return greedy_action
        else:
            return random_action


    def test(self, greedy_action, random_action):
        """
        Args:
            greedy_action(float): action that is outputted from the agent.
            random_action(float): random action variable. Variable used from BaseEnv.sample.
            `test_greedy` default is 1. Set `test_greedy` argument when creating an instance.
        """
        greedy_ratio = np.clip(self.test_greedy,0,1)

        if np.random.rand() > greedy_ratio:
            return greedy_action
        else:
            return random_action

    def value(self):
        return self.func.epsilon



class ConstantFilter(ActionFilter):
    """Constant Filter
    This class allow you to output the agents action with constant epislon.
    """

    def __init__(self, threshold):
        self.th = threshold
        self.value = threshold

    def __call__(self, greedy_action, random_action,
                 step=None, episode=None, epoch=None):
        """
        Args:
            greedy_action(float): action that is outputted from the agent.
            random_action(float): random action variable. Variable used from BaseEnv.sample.
            step(int): total accumulated steps (irrelevent to episodes)
            episode(int): total accumulated episodes (irrelevent to epoch)
            epoch(int): epoch steps
        """
        if np.random.rand() > self.th:
            return greedy_action
        else:
            return random_action

    def test(self,greedy_action, random_action):

        """
        Args:
            greedy_action(float): action that is outputted from the agent.
            random_action(float): random action variable. Variable used from BaseEnv.sample.
        """

        if np.random.rand() > self.th:
            return greedy_action
        else:
            return random_action


    def value(self):
        return self.th






class ActionNoiseFilter(object):
    """Action Noise Filter
    This is the class filter for appending noise to action. Action Noise Filter allows the Agent to explore with noise.
    `__call__``test``value` are required.
    """

    def __call__(self, action,
                 step=None, eposode=None, epoch=None):
        raise NotImplemented

    def test(self, action):
        raise NotImplemented

    def value(self):
        raise NotImplemented





class OUFilter(ActionNoiseFilter):
    """OU Filter
    This class allow is the filter which adds OU(ornstein-uhlenbeck) noise to the action. These are implemented in RL framework.
    This is done by the following equation:

    .. math::

        \epsilon_{t+1}=action+\mathcal{N}_{OU}

    The Default parameters are:
        step_mode = "constant"
        coef = 1
        test_coef = 0
        mu = 0
        theta = 0.15
        sigma = 0.2
    you can specify the step_mode.("step_linear","episode_base" etc.)
    Check the `EpsilonUpdate` class for more details.

    Args:
        **kwargs: parameters such as `coef`,`test_coef`,`theta`,`mu`,`sigma`,`step_mode`
    """

    def __init__(self,**kwargs):

        kwargs["value"] = 1 if "coef" not in kwargs else kwargs["coef"]
        kwargs["step_mode"] = "constant" if "step_mode" not in kwargs else kwargs["step_mode"]

        self.func=EpsilonUpdate(**kwargs)

        self.test_value = kwargs["test_coef"] if "test_coef" in kwargs else 0

        key_list=get_keys(kwargs,"theta","mu","sigma")

        self.noise=OU(**key_list)


    def __call__(self, action,
                 step=None, episode=None, epoch=None):
        return action + self.noise.sample(action) * self.func(step, episode, epoch)

    def test(self,action):
        return action + self.noise.sample(action) * self.test_value

    def value(self):
        """
        Outputs the epsilon value
        """
        return self.func.epsilon


class GPFilter(ActionNoiseFilter):
    """OU Filter
    This class allow is the filter which adds GP(Gaussian Process) noise to the action. These are implemented in RL framework.
    This is done by the following equation:

    .. math::

        \epsilon_{t+1}=action+\mathcal{N}_{GP}


    The Default parameters are:
        step_mode = "constant"
        coef = 1
        test_coef = 0
        mean=0
        std=0.1
    you can specify the step_mode ("step_linear","episode_base" etc.) Check the `EpsilonUpdate` class for more details.

    Args:
        **kwargs: parameters such as `coef`,`test_coef`,`theta`,`mu`,`sigma`,`step_mode`
    """

    def __init__(self, **kawrgs):

        kwargs["value"] = 1 if "value" not in kwargs else kwargs["value"]
        kwargs["step_mode"] = "constant" if "step_mode" not in kwargs else kwargs["step_mode"]

        self.func=EpsilonUpdate(**kwargs)

        self.test_value = kwargs["test_value"] if "test_value" in kwargs else 0

        key_list=get_keys(kwargs,"mean","std")

        self.noise=GP(**key_list)


    def __call__(self, action,
                 step=None, episode=None, epoch=None):
        return action + self.noise.sample(action) * self.func(step, episode, epoch)

    def test(self,action):
        return action + self.noise.sample(action) * self.test_value

    def value(self):
        """
        Outputs the epsilon value
        """
        return self.func.epsilon
