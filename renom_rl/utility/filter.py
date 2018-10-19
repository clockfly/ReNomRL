import numpy as np

class ActionFilter(object):
    """
    This is the parent object of action.
    """

    def __call__(self, action_greedy, action_random,
                 step=None, eposode=None, epoch=None):
        raise NotImplemented

    def test(self, greedy_action, random_action):
        raise NotImplemented

    def value(self):
        raise NotImplemented


class EpsilonGreedyFilter(ActionFilter):
    """

    """

    def __init__(self, **kwargs):

        function_list={
        "step_linear":self._step_linear,
        "episode_inverse":self._episode_inverse,
        }

        self.kwargs=kwargs

        mode = kwargs["mode"] if "mode" in kwargs else "step_linear" # mode exist as args
        mode = mode if mode in function_list else "step_linear" # mode exist as dictionary

        self.func=function_list[mode]

        self.max = kwargs["max"] if "max" in kwargs else 1.0
        self.min = kwargs["min"] if "min" in kwargs else 0.0
        self.initial = kwargs["initial"] if "initial" in kwargs else self.min
        self.greedy_step =  kwargs["greedy_step"] if "greedy_step" in kwargs else 250000
        self.test_greedy = self.kwargs["test_greedy"] if "test_greedy" in kwargs else 0.95

        self.step_size=(self.max-self.min)/self.greedy_step

        self.greedy=self.min



    def __call__(self, greedy_action, random_action,
                 step=None, episode=None, epoch=None):
        """
        Args:
            greedy_action(float)
            random_action(float)
        """

        assert step is not None, \
            "Please give the `step` argument to EpsilonGreedyFilter.__call__."

        greedy_ratio =self.func(step,episode,epoch)
        self.greedy=greedy_ratio

        if np.random.rand() < greedy_ratio:
            return greedy_action
        else:
            return random_action


    def test(self,greedy_action, random_action):

        greedy_ratio = self.test_greedy
        self.greedy=greedy_ratio

        if np.random.rand() < greedy_ratio:
            return greedy_action
        else:
            return random_action


    def value(self):
        return self.greedy

    def _step_linear(self,step, episode, epoch):
        return self._clip(self.step_size*step)


    def _episode_inverse(self,step, episode, epoch):
        return self._clip(1-1/(1+episode*self.kwargs["alpha"]))


    def _clip(self,greedy):
        """
        clipping function
        """
        greedy = np.clip(greedy, self.min, self.max)
        return greedy






class ConstantFilter(ActionFilter):
    """
    """

    def __init__(self, threshold):
        self.th = threshold
        self.value = threshold

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

    def test(self,greedy_action, random_action):

        if np.random.rand() < self.th:
            return greedy_action
        else:
            return random_action


    def value(self):
        return self.value
