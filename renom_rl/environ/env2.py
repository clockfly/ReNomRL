from __future__ import division, print_function, absolute_import
import inspect
import numpy as np


def check_step_method(func, action, state):
    args = inspect.getargspec(func)[0]
    assert len(args) > 1, "Please define argument `action`. Actual argument is {}.".format(args[1:])
    ret = func(action)
    assert ret and len(ret) == 3, "Please define return values `state`, `reward` and `terminal`."
    assert hasattr(ret[0], "shape") and ret[0].shape == state.shape, \
        "Please set the shape of the return value `state` same as `self.state_shape`. Expect {}, actual is {}.".format(
            state.shape, ret[0].shape)


def check_sample_method(func, action, state):
    ret = func()
    assert ret is not None, "Please define return value `random_action`."


def check_reset_method(func, action, state):
    ret = func()
    assert ret is not None, "Please define return value `initial_state`."
    assert hasattr(ret[0], "shape") and ret.shape == state.shape, \
        "Please set the shape of the return value `initial_state` same as `self.state_shape`. Expect {}, actual is {}.".format(
            state.shape, ret.shape)


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
    env=None

    def __init__(self,action_shape=None, state_shape=None, env=None):

        self.env = env
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



    def check_gym(self):
        condition=getattr(self.env, "reset",False) \
                and getattr(self.env, "step", False) \
                and getattr(self.env, "action_space", False)\
                and getattr(self.env, "render", False)

        if condition:
            return True
        else:
            False


    def step(self, action):
        """This method must be overridden.
        This method must accept single action and return `next state`,
        `reward` and `terminal`.
        When overriding this method, don't forget to return above data.

        Returns:
            | (ndarray): Environment's next state.
            | (float): Reward gotten from the transition.
            | (bool): Terminal flag.
            |

        """
        if self.check_gym():
            state,action,terminal,_=self.env.step(int(action))
            return state, action, terminal
        else:
            raise NotImplementedError("Please override `step` method.")

    def sample(self):
        """This method must be overridden.
        This method must return random action.

        Returns:
            (int, ndarray): Sampled action. Its shape must be same as BaseEnv.action_shape.
        """
        if self.check_gym():
            return self.env.action_space.sample()
        else:
            raise NotImplementedError("Please override `sample` method.")

    def reset(self):
        """This method must be overridden.

        Returns:
            (int, ndarray): Initial state. Its shape must be same as BaseEnv.state_shape.
        """
        if self.check_gym():
            return self.env.reset()
        else:
            raise NotImplementedError("Please override `reset` method.")

    def render(self):
        """Rendering method.
        If you want to render environment states, please override this method.
        """
        if self.check_gym():
            return self.env.render(mode="rgb_array")
        else:
            pass

    def terminate(self):
        return False

    def close(self):
        if self.check_gym():
            return self.env.close()
        else:
            pass
