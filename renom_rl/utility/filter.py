import numpy as np
from renom_rl.function.epsilon import Epsilon, E_StepLinear, E_Constant
from renom_rl.function.noise import OU,GP


def check_reframe_epsilon(epsilon,test_epsilon):
    """Check Reframe Epsilon
    This function checks whether the arguments are Epsilon, float, int, or numpy.
    If float, int, or numpy, then they are processed as E_Constant.
    """

    assert isinstance(epsilon,(Epsilon, float, int, np)),
                "epsilon must be an Epsilon object or numerical value (float, int, numpy)"
    assert isinstance(test_epsilon,(Epsilon, float, int, np)),
                "test_epsilon must be an Epsilon object or numerical value (float, int, numpy)"

    epsilon = epsilon if isinstance(epsilon,Epsilon) else E_Constant(epsilon)

    test_epsilon = test_epsilon if isinstance(test_epsilon,Epsilon) else E_Constant(test_epsilon)

    return epsilon , test_epsilon


def check_noise(noise):
    """Check Reframe Epsilon
    This function checks whether argument is noise object.
    """
    assert isinstance(noise,Noise),
                "noise must be a Noise object"

    return noise



class ActionFilter(object):
    """Action Filter
    This is the class of action filter. Action Filter allows the Agent to explore instead of being deterministic in discrete space.
    Class such as epsilon greedy have this as its parent object. Users will get an error if this was not used as the filter.
    `__call__``test``value` are required.
    """

    def __call__(self, action_greedy, action_random,
                 step=None, episode=None, epoch=None):
        raise NotImplemented

    def test(self, greedy_action, random_action):
        raise NotImplemented

    def value(self):
        raise NotImplemented


class EpsilonGreedyFilter(ActionFilter):
    """Epsilon Greedy Filter
    This class allows users to decide action using epsilon greedy filter.
    By specifying the epsilon arugments with Epsilon object, the epsilon will change (or stay the same).
    floats, int, numpy can also be used when constant is required.

    Args:
        epsilon(Epsilon,int,float,numpy): Epsilon during learning. Default is E_StepLinear(1,0,1,25000).
        test_epsilon(Epsilon,int,float,numpy): Epsilon during test. Default is E_Constant(0).

    Examples:
        >>> from renom_rl.utility.filter import EpsilonGreedyFilter
        >>> from renom_rl.function.epsilon import E_EpisodeInverse
        >>>
        >>> a = EpsilonGreedyFilter(0.1,0.1)
        >>> b = EpsilonGreedyFilter(
        ...         epsilon=E_EpisodeInverse(max=1.0,min=0,alpha=0.5),
        ...         test_epsilon=E_EpisodeInverse(max=1.0,min=0,alpha=0.5)
        ...         )
    """

    def __init__(self, epsilon=E_StepLinear(), test_epsilon=E_Constant()):

        epsilon, test_epsilon = check_reframe_epsilon(epsilon,test_epsilon)

        self.func = epsilon
        self.test_func = test_epsilon
        self.epsilon = self.func.epsilon

    def __call__(self, greedy_action, random_action,
                 step=None, episode=None, epoch=None):

        greedy_ratio =self.func(step,episode,epoch)

        self.epsilon = greedy_ratio

        return greedy_action if np.random.rand() > greedy_ratio else random_action


    def test(self, greedy_action, random_action):

        greedy_ratio = self.test_func()

        self.epsilon = greedy_ratio

        return greedy_action if np.random.rand() > greedy_ratio else random_action

    def value(self):
        return self.func.epsilon





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




class AddNoiseFilter(object):
    """Add Noise Filter
    This class allows users to decide action with OU, GP noise etc. with more freedom.
    The epsilon in this class is the coefficient of the noise.
    By specifying the epsilon arugments with Epsilon object, the epsilon will change (or stay the same).
    floats, int, numpy can also be used when constant is required.

    Args:
        epsilon(Epsilon,int,float,numpy): Epsilon during learning. Default is E_StepLinear(1,0,1,25000).
        test_epsilon(Epsilon,int,float,numpy): Epsilon during test. Default is E_Constant(0).
        noise(Noise): Noise Type. Default is OU.

    Examples:
        >>> from renom_rl.utility.filter import AddNoiseFilter
        >>> from renom_rl.function.epsilon import E_EpisodeInverse
        >>>
        >>> a = AddNoiseFilter(0.1,0.1,GP())
        >>> b = EpsilonGreedyFilter(
        ...         epsilon = E_EpisodeInverse(max=1.0,min=0,alpha=0.5),
        ...         test_epsilon = E_EpisodeInverse(max=1.0,min=0,alpha=0.5),
        ...         noise = OU()
        ...         )
    """

    def __init__(self, epsilon = E_StepLinear(), test_epsilon = E_Constant(), noise = OU()):

        epsilon, test_epsilon = check_reframe_epsilon(epsilon,test_epsilon)
        noise = check_noise(noise)

        self.func = epsilon

        self.test_func = test_epsilon

        self.noise = noise

        self.epsilon = self.func.epsilon


    def __call__(self, action,
                 step=None, episode=None, epoch=None):

        epsilon = self.func(step, episode, epoch)

        self.epsilon = epsilon

        return action + self.noise.sample(action) * epsilon

    def test(self,action):

        epsilon = self.test_func()

        self.epsilon = epsilon

        return action + self.noise.sample(action) * epsilon

    def value(self):

        return self.epsilon


class OUFilter(AddNoiseFilter):
    """OUFilter
    This class allows users to use OU Filter. epsilon and test_epsilon are coefficient of the noise.

    Args:
        epsilon(Epsilon,int,float,numpy): Epsilon during learning. Default is E_Constant(1).
        test_epsilon(Epsilon,int,float,numpy): Epsilon during test. Default is E_Constant(0).
        mu(int,float): the value of mu in OU.
        theta(int,float): the value of theta in OU.
        sigma(int,float): the value of sigma in OU.

    Examples:
        >>> from renom_rl.utility.filter import OUFilter
        >>>
        >>> obj = OUFilter(epsilon=0.1, test_epsilon=0, mu=0, theta=0.15, sigma=0.2)
    """

    def __init__(self, epsilon = 1, test_epsilon = 0, mu=0, theta=0.15, sigma=0.2):
        super(OUFilter, self).__init__(epsilon, test_epsilon, OU(mu,theta,sigma))


class GPFilter(AddNoiseFilter):
    """GPFilter
    This class allows users to use GP Filter. epsilon and test_epsilon are coefficient of the noise.

    Args:
        epsilon(Epsilon,int,float,numpy): Epsilon during learning. Default is E_Constant(1).
        test_epsilon(Epsilon,int,float,numpy): Epsilon during test. Default is E_Constant(0).
        mu(int,float): the value of mu in OU.
        theta(int,float): the value of theta in OU.
        sigma(int,float): the value of sigma in OU.

    Examples:
        >>> from renom_rl.utility.filter import GPFilter
        >>>
        >>> obj = GPFilter(epsilon=0.1, test_epsilon=0, mu=0, theta=0.15, sigma=0.2)
    """
    def __init__(self, epsilon = 1, test_epsilon = 0, mean=0, std=0.1):
        super(OUFilter, self).__init__(epsilon, test_epsilon, noise = GP(mean,std)))



    # """OU Filter
    # This class allow is the filter which adds GP(Gaussian Process) noise to the action. These are implemented in RL framework.
    # This is done by the following equation:
    #
    # .. math::
    #
    #     \epsilon_{t+1}=action+\mathcal{N}_{GP}
    #
    #
    # The Default parameters are:
    #     step_mode = "constant"
    #     coef = 1
    #     test_coef = 0
    #     mean=0
    #     std=0.1
    # you can specify the step_mode ("step_linear","episode_base" etc.) Check the `EpsilonUpdate` class for more details.
    #
    # Args:
    #     **kwargs: parameters such as `coef`,`test_coef`,`theta`,`mu`,`sigma`,`step_mode`
    # """
    #
    #
    #
    # """OU Filter
    # This class allow is the filter which adds OU(ornstein-uhlenbeck) noise to the action. These are implemented in RL framework.
    # This is done by the following equation:
    #
    # .. math::
    #
    #     \epsilon_{t+1}=action+\mathcal{N}_{OU}
    #
    # The Default parameters are:
    #     step_mode = "constant"
    #     coef = 1
    #     test_coef = 0
    #     mu = 0
    #     theta = 0.15
    #     sigma = 0.2
    # you can specify the step_mode.("step_linear","episode_base" etc.)
    # Check the `EpsilonUpdate` class for more details.
    #
    # Args:
    #     **kwargs: parameters such as `coef`,`test_coef`,`theta`,`mu`,`sigma`,`step_mode`
    # """


        # """
        # Args:
        #     greedy_action(float): action that is outputted from the agent.
        #     random_action(float): random action variable. Variable used from BaseEnv.sample.
        #     `test_greedy` default is 1. Set `test_greedy` argument when creating an instance.
        # """




# def check_kwargs_exist(kwargs=None,*key):
#     assert all(k in kwargs for k in key), "you have set {} for this step_mode".format(key)
#
# def get_keys(kwargs=None,*keys):
#     res= {}
#     for k in keys:
#       if k in kwargs:
#         res[i]=kwargs[i]
#     return res
#




#
# class ConstantFilter(ActionFilter):
#     """Constant Filter
#     This class allow you to output the agents action with constant epislon.
#     """
#
#     def __init__(self, threshold):
#         self.th = threshold
#         self.value = threshold
#
#     def __call__(self, greedy_action, random_action,
#                  step=None, episode=None, epoch=None):
#         """
#         Args:
#             greedy_action(float): action that is outputted from the agent.
#             random_action(float): random action variable. Variable used from BaseEnv.sample.
#             step(int): total accumulated steps (irrelevent to episodes)
#             episode(int): total accumulated episodes (irrelevent to epoch)
#             epoch(int): epoch steps
#         """
#         if np.random.rand() > self.th:
#             return greedy_action
#         else:
#             return random_action
#
#     def test(self,greedy_action, random_action):
#
#         """
#         Args:
#             greedy_action(float): action that is outputted from the agent.
#             random_action(float): random action variable. Variable used from BaseEnv.sample.
#         """
#
#         if np.random.rand() > self.th:
#             return greedy_action
#         else:
#             return random_action
#
#
#     def value(self):
#         return self.th
