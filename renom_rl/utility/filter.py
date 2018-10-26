import numpy as np
# from renom_rl.function.epsilon import Epsilon, E_StepLinear, E_Constant
# from renom_rl.function.noise import OU,GP


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
    This class allows users to custom and decide action using epsilon greedy filter.
    By specifying the epsilon arugments with Epsilon object, the epsilon will change (or stay the same).
    floats, int, numpy can also be used when constant is required.

    Args:
        epsilon(Epsilon,int,float,numpy): Epsilon during training. Default is E_StepLinear(1,0,1,25000).
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


class StepFilter(EpsilonGreedyFilter):
    """Step Filter
    This class allows users to use Step Filter. Linear function is used, with step as variable, to decrease epsilon.
    The epsilon calculation is as follows:

    :math:

        \epsilon_{t+1}=\epsilon_{0}-\frac{init-min}{epsilon_step} * episode


    Args:

        init(float): initial value. Default is 1.
        min(float): minimum value. Default is 0.
        max(float): maximum value. Default is 1.
        epsilon_step(float): inverse constant of the linear term. Default is 25000.
        test_epsilon(float): epsilon value during test. Default is 0.

    Examples:
        >>> from renom_rl.utility.filter import StepFilter
        >>>
        >>> obj = StepFilter(min=0,max=1,initial=0.5,test_filter=0)
    """
    def __init__(self, initial = 1.0, min = 0.0, max = 1.0,
                    epsilon_step = 25000 ,test_epsilon=0):

        super(StepFilter,self).__init__(E_StepLinear(initial,min,max,epsilon_step),test_epsilon)


class InverseFilter(EpsilonGreedyFilter):
    """Inverse Filter
    This class allows users to use Episode Inverse Filter. Inverse proportion function is used, with episode as variable, to decrease epsilon.
    The epsilon calculation is as follows:

    :math:

        \epsilon_{t+1}=\epsilon_{min}+\frac{init-min}{1+episode * alpha}

    Args:

        init(float): initial value. Default is 1.
        min(float): minimum value. Default is 0.
        max(float): maximum value. Default is 1.
        alpha(float): the coefficient of episode. Default is 1.
        test_epsilon(float): epsilon value during test. Default is 0.

    Examples:
        >>> from renom_rl.utility.filter import InverseFilter
        >>>
        >>> obj = InverseFilter(min=0,max=1,initial=0.5, alpha=1, test_filter=0)
    """
    def __init__(self, initial = 1.0, min = 0.0, max = 1.0,
                    alpha = 1, test_epsilon = 0):

        super(InverseFilter,self).__init__(E_EpisodeInverse(initial,min,max,alpha),test_epsilon)


class ConstantFilter(EpsilonGreedyFilter):
    """Constant Filter
    This class allows users to use Constant Filter. Constant epsilon is used.

    Args:

        epsilon(float): epsilon value during training. Default is 0.1.
        test_epsilon(float): epsilon value during test. Default is 0.

    Examples:
        >>> from renom_rl.utility.filter import InverseFilter
        >>>
        >>> obj = InverseFilter(min=0,max=1,initial=0.5, alpha=1, test_filter=0)
    """

    def __init__(self, epsilon=0.1, test_epsilon = 0):

        super(ConstantFilter,self).__init__(epsilon, test_epsilon)








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




class AddNoiseFilter(ActionNoiseFilter):
    """Add Noise Filter
    This class allows users to decide action with OU, GP noise etc. with more freedom.
    The epsilon in this class is the coefficient of the noise.
    By specifying the epsilon arugments with Epsilon object, the epsilon will change (or stay the same).
    floats, int, numpy can also be used when constant is required.

    Args:
        epsilon(Epsilon,int,float,numpy): Epsilon during training. Default is E_StepLinear(1,0,1,25000).
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
        epsilon(int,float,numpy): Epsilon during training. Default is 1(E_Constant(1)).
        test_epsilon(int,float,numpy): Epsilon during test. Default is 0(E_Constant(0)).
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
        epsilon(int,float,numpy): Epsilon during training. Default is 1(E_Constant(1)).
        test_epsilon(int,float,numpy): Epsilon during test. Default is 0(E_Constant(0)).
        mean(int,float): the mean of GP.
        std(int,float): the standard deviation of GP.

    Examples:
        >>> from renom_rl.utility.filter import GPFilter
        >>>
        >>> obj = GPFilter(epsilon=0.1, test_epsilon=0, mu=0, theta=0.15, sigma=0.2)
    """
    def __init__(self, epsilon = 1, test_epsilon = 0, mean=0, std=0.1):
        super(OUFilter, self).__init__(epsilon, test_epsilon, noise = GP(mean,std)))




class Epsilon(object):
    def __init__(self, initial = 1.0, min = 0.0, max = 1.0):
        self.max = max
        self.min = min
        self.initial = initial
        self.epsilon = min


    def __call__(self,step,episode,epoch):
        pass

    def _clip(self,greedy):
        """
        clipping function
        """
        greedy = np.clip(greedy, self.min, self.max)
        return greedy


class EpsilonSL(Epsilon):
    """EpsilonSL (Epsilon Step Linear)
    """
    def __init__(self, initial = 1.0, min = 0.0, max = 1.0,
                    epsilon_step = 25000 ):

        super(StepLinear,self).__init__(initial,min,max)

        self.test_epsilon = test_epsilon
        self.epsilon_step =  epsilon_step
        self.step_size = (max-min)/epsilon_step

    def __call__(self,step,episode,epoch):
        self.epsilon = self._clip(self.initial-self.step_size*step)
        return self.epsilon


class EpsilonEI(Epsilon):
    """EpsilonEI (Epsilon Episode Inverse)
    """
    def __init__(self, initial = 1.0, min = 0.0, max = 1.0,
                    alpha = 1 ):

        super(StepLinear,self).__init__(initial,min,max)

        self.alpha = alpha

    def __call__(self,step,episode,epoch):
        self.epsilon = self._clip(self.min+(self.initial-self.min)/(1+episode*self.alpha))
        return self.epsilon


class EpsilonC(Epsilon):
    """EpsilonC (Epsilon Constant)
    """
    def __init__(self,epsilon=0.0):
        self.epsilon=epsilon

    def call(self,step, episode, epoch):
        return np.clip(self.epsilon,0,1)

class Noise(object):
    def __init__(self):
        pass

    def sample(self, action):
        raise NotImplemented





class OU(Noise):
    """
    DDPG paper ornstein-uhlenbeck noise parameters are theta=0.15, sigma=0.2
    """

    def __init__(self, mu=0, theta=0.15, sigma=0.2):
        self.noise_type = CONTINUOUS
        self.mu = mu
        self.theta = theta
        self.sigma = sigma

    def sample(self, action):
        shape = getattr(self.mu, 'shape', [1, ])
        return self.theta * (self.mu - action) + self.sigma \
            * np.random.randn(*shape)


class GP(object):

    def __init__(self, mean=0, std=0.1):
        self.noise_type = CONTINUOUS
        self._mean = mean
        self._std = std

    def sample(self, action):
    # def samplse(self, action):
        shape = getattr(action, 'shape', [1, ])
        return self._mean + self._std * np.random.randn(*shape)
