from __future__ import division, print_function, absolute_import
import inspect
import numpy as np


def check_step_method(func, action, state):
    args = inspect.getargspec(func)[0]
    assert len(args) > 1, "Please define argument ``action``. Actual argument is {}.".format(
        args[1:])
    ret = func(action)
    assert ret and len(
        ret) == 3, "Please define return values ``state``, ``reward`` and ``terminal``."
    assert hasattr(ret[0], "shape") and ret[0].shape == state.shape, \
        "Please set the shape of the return value ``state`` same as ``self.state_shape``. Expect {}, actual is {}.".format(
            state.shape, ret[0].shape)


def check_sample_method(func, action, state):
    ret = func()
    assert ret is not None, "Please define return value ``random_action``."


def check_reset_method(func, action, state):
    ret = func()
    assert ret is not None, "Please define return value ``initial_state``."
    assert hasattr(ret[0], "shape") and ret.shape == state.shape, \
        "Please set the shape of the return value ``initial_state`` same as ``self.state_shape``. Expect {}, actual is {}.".format(
            state.shape, ret.shape)


class BaseEnv(object):
    """Base class of environment.
    The methods ``step``, ``reset`` and ``sample`` must be overridden.
    Users can override ``start``, ``epoch``, ``epoch_step``, ``close``, ``test_start``, ``test_epoch_step``, ``test_close``.
    If  ``test_start``, ``test_epoch_step``, ``test_close`` are not defined, then they will use the same function as ``start``, ``epoch_step``, ``close``.
    Note that these functions are only used to execute, thus arguments can not be implemented (excluding ``reset``).

    Example:
        >>> import numpy as np
        >>> from renom_rl import BaseEnv
        >>> class CustomEnv(BaseEnv):
        ...    def __init__(self):
        ...         action_shape = (5, )
        ...         state_shape = (86, 86)
        ...
        ...     def step(self, action):
        ...         state, reward, terminal = func(action)
        ...         return state, reward, terminal
        ...
        ...     def sample(self):
        ...         return self.step(np.random.randInt(0, 5))[0]
        ...
        ...     def reset(self):
        ...         initial_state=func.reset()
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
        This method must accept single action and return next state,
        reward and terminal.
        When overriding this method, don't forget to return above data.

        Returns:
            | (ndarray): Environment's next state.
            | (float): Reward gotten from the transition.
            | (bool): Terminal flag.
            |

        """
        raise NotImplementedError("Please override ``step`` method.")

    def sample(self):
        """This method must be overridden.
        This method must return random action.

        Returns:
            (int, ndarray): Sampled action. Its shape must be same as BaseEnv.action_shape.
        """
        raise NotImplementedError("Please override ``sample`` method.")

    def reset(self):
        """This method must be overridden.

        Returns:
            (int, ndarray): Initial state. Its shape must be same as BaseEnv.state_shape.
        """
        raise NotImplementedError("Please override ``reset`` method.")

    # change def()
    def terminate(self):
        """This is optional.
        In some cases, users want to terminate learning for certain conditions.
        By overriding this function, users will able to terminate the learning process
        once a certain condition is met. The return value is False by default.
        """
        return False

    def start(self):
        """ This is optional.
        This method will be called when fit function starts.
        This will execute after ``reset``.
        """
        pass

    def epoch(self):
        """ This is optional.
        This method will be called when epoch is updated.
        This will execute after ``reset``.
        """
        pass

    def epoch_step(self):
        """ This is optional.
        This method will be called every step.
        This will execute after ``step``.
        """
        pass

    def close(self):
        """ This is optional.
        This method will be called when fit is closed.
        """
        pass

    def test_start(self):
        """ This is optional.
        This method will be called when test is starting.
        This will execute after ``reset``.
        """
        self.start()

    def test_epoch_step(self):
        """ This is optional.
        This method will be called every step.
        This will execute after ``step``.
        """
        self.epoch_step()

    def test_close(self):
        """ This is optional.
        This method will be called when test is done.
        """
        self.close()
