import numpy as np
from .epsilon import EpsilonUpdate

class ActionFilter(object):
    """

    """
    def __init__(self, filter="epsilon_greedy", **kwargs):

        self.f_dictionary={
            "epsilon_greedy":EpsilonGreedy,
        }

        self.obj=self.f_dictionary[filter](**kwargs)
        self.filter=filter

        # set "step_0" by default

    def __call__(self, action, env, **var):
        return self.obj(action,env,**var)

    def value(self):
        return self.obj.value()




class EpsilonGreedy(object):
    """
    """
    def __init__(self,**kwargs):

        self.epsilon=EpsilonUpdate(**kwargs)

        self.kwargs=kwargs


    def __call__(self,action,env,**var):

        greedy=self.epsilon.update(**var)

        if greedy > np.random.rand():  # and state is not None:
            action_f = action
        else:
            action_f = env.sample()

        return action_f


    def test(self,action, env, **var):

        greedy = var["greedy"]

        if greedy > np.random.rand():  # and state is not None:
            action_f = action
        else:
            action_f = env.sample()

        return action_f


    def value(self):

        return self.epsilon.value()
