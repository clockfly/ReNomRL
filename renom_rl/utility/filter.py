import numpy as np

class ActionFilter(object):
    """Action Filter
    This is the object of action filter. Action Filter allows the Angent to explore instead of being deterministic.
    Objects such as epsilon greedy have this as its parent object. Users will get an error if this was not used as the filter.
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
    This objects allow you to change the epsilon function. These are implemented in RL framework.
    By specifying the greedy mode ("step_linear","episode_base" etc.), you can change the epsilon mode.
    All functions are in the range of min and max.

    Args:
        mode (string): mode of learning.
        **kwargs: parameters such as `min`, `max`, `greedy_step`, `episode`, `alpha`, `test_greedy`

    Example:
        >>> update = EpsilonUpdate(mode="step_linear",initial=min_greedy,min=min_greedy,max=max_greedy,greedy_step=greedy_step)
        >>> greedy = update.init()
        >>> greedy = update.push()

    what is `test_greedy`:
        test

    Mode:

        :mode: step_linear
        :required variables: `initial`,`min`,`max`,`greedy_step`
        :function:
            .. math::

                \epsilon_{t+1}=\epsilon_{t}+\frac{(max-min)}{greedy_step}

        :note: t = step.

        ---------

        :mode: episode_inverse
        :required variables: `initial`,`min`,`max`,`alpha`
        :function:
            .. math::

                \epsilon_{t+1}= 1 - \frac{1}{(1+ episode * alpa)}

        ---------
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
            greedy_action(float): action that is outputted from the agent.
            random_action(float): random action variable. Variable used from BaseEnv.sample.
            step: total accumulated steps (irrelevent to episodes)
            episode: total accumulated episodes (irrelevent to epoch)
            epoch: epoch steps
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
        """
        Args:
            greedy_action(float): action that is outputted from the agent.
            random_action(float): random action variable. Variable used from BaseEnv.sample.
            `test_greedy` argument is required.
        """
        greedy_ratio = self.test_greedy
        self.greedy=greedy_ratio

        if np.random.rand() < greedy_ratio:
            return greedy_action
        else:
            return random_action


    def value(self):
        """
        Outputs the epsilon value
        """
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
    """Constant Filter
    This objects allow you to output the agents action with constant epislon.
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
            step: total accumulated steps (irrelevent to episodes)
            episode: total accumulated episodes (irrelevent to epoch)
            epoch: epoch steps
        """
        if np.random.rand() < self.th:
            return greedy_action
        else:
            return random_action

    def test(self,greedy_action, random_action):

        """
        Args:
            greedy_action(float): action that is outputted from the agent.
            random_action(float): random action variable. Variable used from BaseEnv.sample.
        """

        if np.random.rand() < self.th:
            return greedy_action
        else:
            return random_action


    def value(self):
        return self.value
