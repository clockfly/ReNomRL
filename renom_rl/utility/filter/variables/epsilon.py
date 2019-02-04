import numpy as np


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
