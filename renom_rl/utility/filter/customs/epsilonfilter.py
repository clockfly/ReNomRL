import numpy as np
from ..filter_base import EpsilonGreedyFilter
from ..variables import EpsilonSL, EpsilonEI


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
