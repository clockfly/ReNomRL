import numpy as np
from renom import Node


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
        raise NotImplemented

    def test(self, greedy_action, random_action):
        raise NotImplemented

    def value(self):
        raise NotImplemented


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


class EpsilonSLFilter(EpsilonGreedyFilter):
    """
    **Epsilon Step Linear Filter**

    This class allows users to use Epsilon Greedy with step linear change.
    Linear function is used, with step as variable, to decrease epsilon.
    The epsilon calculation is as follows:

    :math:

        .. math::

            \\epsilon &= \\epsilon_{0} - \\frac{init-min}{epsilon\_step} step

    Args:

        init(float): initial value.
        min(float): minimum value.
        max(float): maximum value.
        epsilon_step(float): inverse constant of the linear term.
        test_epsilon(float): epsilon value during test.

    Examples:
        >>> from renom_rl.utility.filter import EpsilonSLFilter
        >>>
        >>> obj = EpsilonSLFilter(min=0,max=1,initial=0.5,test_filter=0)
    """

    def __init__(self, initial=1.0, min=0.0, max=1.0,
                 epsilon_step=25000, test_epsilon=0):

        epsilon = EpsilonSL(initial, min, max, epsilon_step)

        super(EpsilonSLFilter, self).__init__(epsilon, test_epsilon)


class EpsilonEIFilter(EpsilonGreedyFilter):
    """
    **Epsilon Episode Inverse Filter**

    This class allows users to use Epsilon Greedy with Epsilon episode inverse change.
    Inverse proportion function is used, with episode as variable, to decrease epsilon.
    The epsilon calculation is as follows:

    :math:

        .. math::

            \\epsilon &= \\epsilon_{min} + \\frac{init-min}{1+episode * alpha}

    Args:

        init(float): initial value.
        min(float): minimum value.
        max(float): maximum value.
        alpha(float): the coefficient of episode.
        test_epsilon(float): epsilon value during test.

    Examples:
        >>> from renom_rl.utility.filter import EpsilonEIFilter
        >>>
        >>> obj = EpsilonEIFilter(min=0,max=1,initial=0.5, alpha=1, test_filter=0)
    """

    def __init__(self, initial=1.0, min=0.0, max=1.0,
                 alpha=1, test_epsilon=0):

        epsilon = EpsilonEI(initial, min, max, alpha)

        super(EpsilonEIFilter, self).__init__(epsilon, test_epsilon)


class EpsilonCFilter(EpsilonGreedyFilter):
    """
    **Constant Filter**

    This class allows users to use Constant Filter. Constant epsilon is used.

    Args:

        epsilon(float): epsilon value during training.
        test_epsilon(float): epsilon value during test.

    Examples:
        >>> from renom_rl.utility.filter import EpsilonCFilter
        >>>
        >>> obj = EpsilonCFilter(epsilon=0.1, test_filter=0)
    """

    def __init__(self, epsilon=0.1, test_epsilon=0):

        super(EpsilonCFilter, self).__init__(epsilon, test_epsilon)


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
        raise NotImplemented

    def value(self):
        raise NotImplemented

    def sample(self):
        raise NotImplemented


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

        return self.noise_value


class OUFilter(AddNoiseFilter):
    """
    **OUFilter**

    This class allows users to use OU Filter. epsilon and test_epsilon are coefficient of the noise.

    Args:
        epsilon(float): Epsilon during training.
        test_epsilon(float): Epsilon during test.
        mu(float): the value of mu in OU.
        theta(float): the value of theta in OU.
        sigma(float): the value of sigma in OU.

    Examples:
        >>> from renom_rl.utility.filter import OUFilter
        >>>
        >>> obj = OUFilter(epsilon=0.1, test_epsilon=0, mu=0, theta=0.15, sigma=0.2)
    """

    def __init__(self, epsilon=1, test_epsilon=0, mu=0, theta=0.15, sigma=0.2):

        noise = OU(mu, theta, sigma)

        super(OUFilter, self).__init__(epsilon, test_epsilon, noise)


class GPFilter(AddNoiseFilter):
    """
    **GPFilter**

    This class allows users to use GP Filter. epsilon and test_epsilon are coefficient of the noise.

    Args:
        epsilon(float): Epsilon during training.
        test_epsilon(float): Epsilon during test.
        mean(float): the mean of GP.
        std(float): the standard deviation of GP.

    Examples:
        >>> from renom_rl.utility.filter import GPFilter
        >>>
        >>> obj = GPFilter(epsilon=0.1, test_epsilon=0, mu=0, theta=0.15, sigma=0.2)
    """

    def __init__(self, epsilon=1, test_epsilon=0, mean=0, std=0.1):

        noise = GP(mean, std)

        super(GPFilter, self).__init__(epsilon, test_epsilon, noise)


class NoNoiseFilter(AddNoiseFilter):
    """
    **No Noise Filter**

    This class allows No Noise as filter.

    Examples:
        >>> from renom_rl.utility.filter import GPFilter
        >>>
        >>> obj = NoNoiseFilter()
    """

    def __init__(self):

        super(NoNoiseFilter, self).__init__(0, 0)


class Epsilon(object):
    """
    **Epsilon**

    Base Class of Epsilon.
    You can use this object to modify the epsilon change.
    This class uses ``__call__`` to update epsilon variable.
    There is also ``_clip`` function which is used to clip the epsilon between min max value.
    """

    def __init__(self, initial=1.0, min=0.0, max=1.0):
        self.max = max
        self.min = min
        self.initial = initial
        self.epsilon = min

    def __call__(self, step, episode, epoch):
        pass

    def _clip(self, greedy):
        """
        clipping function
        """
        greedy = np.clip(greedy, self.min, self.max)
        return greedy


class EpsilonSL(Epsilon):
    """
    **Epsilon Step Linear**

    Linear function is used, with step as variable, to decrease epsilon.
    The epsilon calculation is as follows:

    :math:

        .. math::

            \\epsilon=\\epsilon_{0}-\\frac{init-min}{epsilon\_step} * episode

    Args:

        init(float): initial value.
        min(float): minimum value.
        max(float): maximum value.
        epsilon_step(float): inverse constant of the linear term.

    """

    def __init__(self, initial=1.0, min=0.0, max=1.0,
                 epsilon_step=25000):

        super(EpsilonSL, self).__init__(initial, min, max)

        self.epsilon_step = epsilon_step
        self.step_size = (max-min)/epsilon_step

    def __call__(self, step, episode, epoch):
        self.epsilon = self._clip(self.initial-self.step_size*step)
        return self.epsilon


class EpsilonEI(Epsilon):
    """
    **Epsilon Episode Inverse**

    Inverse proportion function is used, with episode as variable, to decrease epsilon.
    The epsilon calculation is as follows:

    :math:

        .. math::

            \\epsilon=\\epsilon_{min}+\\frac{init-min}{1+episode * alpha}

    Args:

        init(float): initial value.
        min(float): minimum value.
        max(float): maximum value.
        alpha(float): the coefficient of episode.

    """

    def __init__(self, initial=1.0, min=0.0, max=1.0,
                 alpha=1):

        super(EpsilonEI, self).__init__(initial, min, max)

        self.alpha = alpha

    def __call__(self, step, episode, epoch):
        self.epsilon = self._clip(self.min+(self.initial-self.min)/(1+episode*self.alpha))
        return self.epsilon


class EpsilonC(Epsilon):
    """
    **Epsilon Constant**

    This class allows users to use Constant Filter. Constant epsilon is used.
    """

    def __init__(self, epsilon=0.0):
        self.epsilon = epsilon

    def __call__(self, step=0, episode=0, epoch=0):

        return np.clip(self.epsilon, 0, 1)


class Noise(object):
    def __init__(self):
        pass

    def sample(self, action):
        raise NotImplemented


class OU(Noise):
    """
    **OU (Ornstein-Uhlenbeck)**

    DDPG paper ornstein-uhlenbeck noise parameters are theta=0.15, sigma=0.2
    """

    def __init__(self, mu=0, theta=0.15, sigma=0.2):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma

    def sample(self, action):
        shape = getattr(self.mu, 'shape', [1, ])
        return self.theta * (self.mu - action) + self.sigma \
            * np.random.randn(*shape)


class GP(Noise):
    """
    **GP (Gaussian Noise)**

    Gaussian Noise.
    """

    def __init__(self, mean=0, std=0.1):
        self._mean = mean
        self._std = std

    def sample(self, action):
        # def samplse(self, action):
        shape = getattr(action, 'shape', [1, ])
        return self._mean + self._std * np.random.randn(*shape)


class DiscreteNodeChooser(object):
    """
    Base class of chosing action nodes in discrete action space.
    """

    def __call__(self):
        raise "please override this function"

    def _trasform_node_2_numpy(self, node_var):
        """
        this function changes node variables to numpy
        """

        if isinstance(node_var, Node):
            node_var = node_var.as_ndarray()

        assert len(node_var.shape) > 1, "The node_var must be more than 2D"

        return node_var


class MaxNodeChooser(DiscreteNodeChooser):
    """
    **Max Node Chooser**

    Chooses max node index. Returns 1D numpy list.
    If the length is 1, it returns int.
    """

    def __call__(self, node_var):

        node_var = self._trasform_node_2_numpy(node_var)

        max_list = np.argmax(node_var, axis=1)
        if len(max_list) == 1:
            return int(max_list)
        else:
            return max_list


class ProbNodeChooser(DiscreteNodeChooser):
    """
    **Probability Node Chooser**

    Chooses node index based on its output probability. Returns 1D numpy list.
    If the length is 1, it returns int.
    """

    def __call__(self, node_var):

        node_var = self._trasform_node_2_numpy(node_var)

        norm = np.sum(node_var, axis=1).reshape((-1, 1))

        node_norm = node_var/norm

        prob_list = np.array([np.random.choice(len(n), 1, p=n) for n in node_norm])

        if len(prob_list) == 1:
            return int(prob_list)
        else:
            return prob_list
