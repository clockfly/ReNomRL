import numpy as np
from .variables import Epsilon, EpsilonC, EpsilonSL, Noise, OU


def check_reframe_epsilon(epsilon, test_epsilon):
    """Check Reframe Epsilon
    This function checks whether the arguments are Epsilon, float.
    If float, int, or numpy, then they are processed as EpsilonC.
    """

    assert isinstance(epsilon, (Epsilon, float, int, np)),\
        "epsilon must be an Epsilon object or numerical value (float, int, numpy)"
    assert isinstance(test_epsilon, (EpsilonC, float, int, np)),\
        "test_epsilon must be an EpsilonC object or numerical value (float, int, numpy)"

    epsilon = epsilon if isinstance(epsilon, Epsilon) else EpsilonC(epsilon)

    test_epsilon = test_epsilon if isinstance(test_epsilon, EpsilonC) else EpsilonC(test_epsilon)

    return epsilon, test_epsilon


def check_noise(noise):
    """Check Noise
    This function checks if the argument is a Noise object or not.
    """
    assert isinstance(noise, Noise),\
        "noise must be a Noise object"

    return noise


class ActionFilter(object):
    """
    **Action Filter**

    This is the class of action filter. Action Filter allows the Agent to explore instead of being deterministic in discrete space.
    Class such as epsilon greedy have this as its parent object. Users will get an error if this was not used as the filter.
    ``__call__``, ``test``, ``value`` functions are required.
    """

    def __call__(self, action_greedy, action_random,
                 step=None, episode=None, epoch=None):
        raise NotImplementedError

    def test(self, greedy_action, random_action):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError


class EpsilonGreedyFilter(ActionFilter):
    """
    **Epsilon Greedy Filter**

    This class allows users to custom and decide action using epsilon greedy filter.
    By specifying the epsilon arugments with Epsilon class, the epsilon will change (or stay the same).
    Floats can also be used when constant is required.

    Args:
        epsilon(Epsilon,float): Epsilon during training. Default is ``EpsilonSL(1,0,1,25000)``.
        test_epsilon(EpsilonC,float): Epsilon during test. Default is ``EpsilonC(0)``.

    Examples:
        >>> from renom_rl.utility.filter import EpsilonGreedyFilter, EpsilonEI, EpsilonC
        >>>
        >>>
        >>> a = EpsilonGreedyFilter(0.1,0.1)
        >>> b = EpsilonGreedyFilter(
        ...         epsilon=EpsilonEI(max=1.0,min=0,alpha=0.5),
        ...         test_epsilon=EpsilonC(0.1),
        ...         )
    """

    def __init__(self, epsilon=None, test_epsilon=None):

        # epsilon default
        epsilon = epsilon if epsilon is not None else EpsilonSL()
        test_epsilon = test_epsilon if test_epsilon is not None else EpsilonC()

        # checking and changing datatype
        epsilon, test_epsilon = check_reframe_epsilon(epsilon, test_epsilon)

        self.func = epsilon
        self.test_func = test_epsilon
        self.epsilon = self.func.epsilon

    def __call__(self, greedy_action, random_action,
                 step=None, episode=None, epoch=None):

        greedy_ratio = self.func(step, episode, epoch)

        self.epsilon = greedy_ratio

        return greedy_action if np.random.rand() > greedy_ratio else random_action

    def test(self, greedy_action, random_action):

        greedy_ratio = self.test_func()

        self.epsilon = greedy_ratio

        return greedy_action if np.random.rand() > greedy_ratio else random_action

    def value(self):
        return self.func.epsilon





class ActionNoiseFilter(object):
    """
    **Action Noise Filter**

    This is the class filter for appending noise to action. Action Noise Filter allows the Agent to explore with noise.
    ``__call__``, ``test``, ``value`` are required.
    """

    def __call__(self, action,
                 step=None, eposode=None, epoch=None):
        raise NotImplemented

    def test(self, action):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError


class AddNoiseFilter(ActionNoiseFilter):
    """
    **Add Noise Filter**

    This class allows users to decide action with OU, GP noise etc. with more freedom.
    The epsilon in this class is the coefficient of the noise.
    By specifying the epsilon arugments with Epsilon object, the epsilon will change (or stay the same).
    floats, int, numpy can also be used when constant is required.

    Args:
        epsilon(Epsilon,float): Epsilon during training. Default is ``EpsilonSL(1,0,1,25000)``.
        test_epsilon(EpsilonC,float): Epsilon during test. Default is ``EpsilonC(0)``.
        noise(Noise): Noise Type. Default is ``OU()``.

    Examples:
        >>> from renom_rl.utility.filter import AddNoiseFilter
        >>> from renom_rl.function.epsilon import EpsilonEI, EpsilonC, OU
        >>>
        >>> a = AddNoiseFilter(0.1,0.1,GP())
        >>> b = EpsilonGreedyFilter(
        ...         epsilon = EpsilonEI(max=1.0,min=0,alpha=0.5),
        ...         test_epsilon = EpsilonC(epsilon=0.1),
        ...         noise = OU(mu=0.2)
        ...         )
    """

    def __init__(self, epsilon=None, test_epsilon=None, noise=None):

        epsilon = epsilon if epsilon is not None else EpsilonC()
        test_epsilon = test_epsilon if test_epsilon is not None else EpsilonC()
        noise = noise if noise is not None else OU()

        epsilon, test_epsilon = check_reframe_epsilon(epsilon, test_epsilon)
        noise = check_noise(noise)

        self.func = epsilon

        self.test_func = test_epsilon

        self.noise = noise

        self.epsilon = self.func.epsilon

        self.noise_value = 0

    def __call__(self, action,
                 step=None, episode=None, epoch=None):

        epsilon = self.func(step, episode, epoch)

        self.epsilon = epsilon
        self.noise_value = self.noise.sample(action) * epsilon

        return action + self.noise_value

    def test(self, action):

        epsilon = self.test_func()

        self.epsilon = epsilon
        self.noise_value = self.noise.sample(action) * epsilon

        return action + self.noise_value

    def value(self):

        return self.epsilon

    def sample(self):

        return self.noise_value if not isinstance(self.epsilon, EpsilonC) else 0
