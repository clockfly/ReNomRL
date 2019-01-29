import numpy as np
from ..filter_base import AddNoiseFilter
from ..variables import OU, GP

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
