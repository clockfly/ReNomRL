from __future__ import division, print_function, absolute_import
import inspect
import numpy as np


def check_step_method(func, action, state):
    args = inspect.getargspec(func)[0]
    assert len(args) > 1, "Please define argument `action`. Actual argument is {}.".format(args[1:])
    ret = func(action)
    assert ret and len(ret) == 3, "Please define return values `state`, `reward` and `terminal`."
    assert hasattr(ret[0], "shape") and ret[0].shape == state.shape, \
        "Please set the shape of the return value `state` same as `self.state_shape`."


def check_sample_method(func, action, state):
    ret = func()
    assert ret is not None, "Please define return value `random_action`."


def check_reset_method(func, action, state):
    ret = func()
    assert ret is not None, "Please define return value `initial_state`."
    assert hasattr(ret[0], "shape") and ret.shape == state.shape, \
        "Please set the shape of the return value `initial_state` same as `self.state_shape`."


class BaseEnv(object):
    """Base class of environment.
    The methods `step`, `reset` and `sample` must be overridden.

    Example:
        >>> import numpy as np
        >>> from renom_rl import BaseEnv
        >>> class CustomEnv(BaseEnv):
        ...    def __init__(self):
        ...         action_shape = (5, )
        ...         state_shape = (86, 86)
        ...
        ...     def step(self, action):
        ...         return state, reward, terminal
        ...
        ...     def sample(self):
        ...         return self.step(np.random.randInt(0, 5))[0]
        ...
        ...     def reset(self):
        ...         return initial_state
        ...
    """

    action_shape = None
    state_shape = None

    def __init__(self, action_shape=None, state_shape=None):
        self.action_shape = self.action_shape if self.action_shape else action_shape
        self.state_shape = self.state_shape if self.state_shape else state_shape

        assert self.action_shape, "The field `self.action_shape` must be specified."
        assert self.state_shape, "The field `self.state_shape` must be specified."

        check_reset_method(self.reset,
                           self.sample(),
                           np.zeros(self.state_shape, dtype=np.float32))

        check_step_method(self.step,
                          self.sample(),
                          np.zeros(self.state_shape, dtype=np.float32))

        check_sample_method(self.sample,
                            self.sample(),
                            np.zeros(self.state_shape, dtype=np.float32))

    def step(self, action):
        """This method must be overridden.
        This method must accept single action and return `next state`,
        `reward` and `terminal`.
        When overriding this method, don't forget to return above data.
        """
        raise NotImplementedError("Please override `step` method.")

    def sample(self):
        """This method must be overridden.
        This method must return random action.
        """
        raise NotImplementedError("Please override `sample` method.")

    def reset(self):
        """This method must be overridden.
        """
        raise NotImplementedError("Please override `reset` method.")

    def render(self):
        """
        """
        pass
