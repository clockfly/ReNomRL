import numpy as np


def check_kwargs_exist(kwargs=None,*key):
    assert all(k in kwargs for k in key), "you have set {} for this step_mode".format(key)



class EpsilonUpdate(object):
    """ EpsilonUpdate Class
    This class allow you to change the epsilon value. These are implemented in filter classes.
    By specifying the greedy `step_mode` ("step_linear","episode_base" etc.), you can change the epsilon step_mode.
    All functions are in the range of min and max.

    Args:
        step_mode (string): step mode of how the epsilon increase.
        **kwargs: parameters such as min, max, epsilon_step, episode, alpha

    Example:
        >>> update = EpsilonUpdate(step_mode="step_linear",initial=min_greedy,min=min_greedy,max=max_greedy,epsilon_step=epsilon_step)
        >>> greedy = update.init()
        >>> greedy = update.push()


    step_mode:

        :step_mode: step_linear
        :required variables: "initial","min","max","epsilon_step"
        :function:
            .. math::

                \epsilon_{t+1} = \epsilon_{t}+\frac{(max-min)}{epsilon_step}

        :note: t = step.

        ---------

        :step_mode: episode_base
        :required variables: "initial","min","max","alpa","episode"
        :function:
            .. math::

                \epsilon_{t+1} = 1 - \frac{1}{(1+ episode * alpa)}

        ---------

        :step_mode: constant
        :required variables: "value"
        :function:
            .. math::

                \epsilon = value

        ---------



    """
    def __init__(self,**kwargs):

        function_list={
        "step_linear":self._step_linear,
        "episode_inverse":self._episode_inverse,
        "constant":self._constant,
        }

        step_mode = kwargs["step_mode"] if "step_mode" in kwargs else "step_linear"
        assert step_mode in function_list, "there is no {} as step_mode.".format(step_mode)

        self.func=function_list[step_mode]


        self.max = kwargs["max"] if "max" in kwargs else 1.0
        self.min = kwargs["min"] if "min" in kwargs else 0.0
        self.initial = kwargs["initial"] if "initial" in kwargs else self.max
        self.test_epsilon = kwargs["test_epsilon"] if "test_epsilon" in kwargs else 0.0
        self.epsilon=self.max
        print(self.epsilon)


        if step_mode =="step_linear":
            self.epsilon_step =  kwargs["epsilon_step"] if "epsilon_step" in kwargs else 20000
            self.step_size = (self.max-self.min)/self.epsilon_step

        elif step_mode=="episode_inverse":
            check_kwargs_exist(kwargs,"alpha")
            self.alpha =  kwargs["alpha"]

        elif step_mode=="constant":
            check_kwargs_exist(kwargs,"value")
            self.constant =  kwargs["value"]


    def __call__(self,step,episode,epoch):
        self.epsilon = self.func(step,episode,epoch)
        return self.epsilon

    def _step_linear(self,step, episode, epoch):
        return self._clip(self.initial-self.step_size*step)

    def _episode_inverse(self,step, episode, epoch):
        return self._clip(self.min+(self.initial-self.min)/(1+episode*self.alpha))

    def _constant(self,step, episode, epoch):
        return np.clip(self.constant,0,1)

    def _clip(self,greedy):
        """
        clipping function
        """
        greedy = np.clip(greedy, self.min, self.max)
        return greedy
