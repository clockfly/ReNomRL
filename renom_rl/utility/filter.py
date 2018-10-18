import numpy as np

class ActionFilter(object):
    """

    """

    def __call__(self, action_greedy, action_random,
                 step=None, eposode=None, epoch=None):
        raise NotImplemented


class EpsilonGreedyFilter(ActionFilter):
    """
    """

    def __init__(self, initial, min, max, step):
        self.initial = initial
        self.max = max
        self.min = min
        self.step_size = (max - min) / step

    def __call__(self, greedy_action, random_action,
                 step=None, episode=None, epoch=None):
        """
        Args:
            greedy_action(float)
            random_action(float)
        """
        assert step is not None, \
            "Please give the `step` argument to EpsilonGreedyFilter.__call__."
        greedy_ratio = np.clip(self.step_size * step + self.min, self.min, self.max)
        if np.random.rand() < greedy_ratio:
            return greedy_action
        else:
            return random_action


class ConstantFilter(ActionFilter):
    """
    """

    def __init__(self, threshold):
        self.th = threshold

    def __call__(self, greedy_action, random_action,
                 step=None, episode=None, epoch=None):
        """
        Args:
            action_greedy(float)
            action_random(float)
        """
        if np.random.rand() < self.th:
            return greedy_action
        else:
            return random_action
