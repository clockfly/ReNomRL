import numpy as np

class Epsilon(object):
    def __init__(self, initial = 1.0, min = 0.0, max = 1.0):
        self.max = max
        self.min = min
        self.initial = initial
        self.epsilon = min


    def __call__(self,step,episode,epoch):
        pass

    def _clip(self,greedy):
        """
        clipping function
        """
        greedy = np.clip(greedy, self.min, self.max)
        return greedy


class E_StepLinear(Epsilon):
    def __init__(self, initial = 1.0, min = 0.0, max = 1.0,
                    epsilon_step = 25000 ):

        super(StepLinear,self).__init__(initial,min,max)

        self.test_epsilon = test_epsilon
        self.epsilon_step =  epsilon_step
        self.step_size = (max-min)/epsilon_step

    def __call__(self,step,episode,epoch):
        self.epsilon = self._clip(self.initial-self.step_size*step)
        return self.epsilon


class E_EpisodeInverse(Epsilon):
    def __init__(self, initial = 1.0, min = 0.0, max = 1.0,
                    alpha = 1 ):

        super(StepLinear,self).__init__(initial,min,max)

        self.alpha = alpha

    def __call__(self,step,episode,epoch):
        self.epsilon = self._clip(self.min+(self.initial-self.min)/(1+episode*self.alpha))
        return self.epsilon


class E_Constant(Epsilon):
    def __init__(self,epsilon=0.0):
        self.epsilon=epsilon

    def call(self,step, episode, epoch):
        return np.clip(self.epsilon,0,1)



# class EpsilonUpdate(object):
#     """ EpsilonUpdate Class
#     This class allow you to change the epsilon value. These are implemented in filter classes.
#     By specifying the greedy `step_mode` ("step_linear","episode_base" etc.), you can change the epsilon step_mode.
#     All functions are in the range of min and max.
#
#     Args:
#         step_mode (string): step mode of how the epsilon increase.
#         **kwargs: parameters such as min, max, epsilon_step, episode, alpha
#
#     Example:
#         >>> update = EpsilonUpdate(step_mode="step_linear",initial=min_greedy,min=min_greedy,max=max_greedy,epsilon_step=epsilon_step)
#         >>> greedy = update.init()
#         >>> greedy = update.push()
#
#
#     step_mode:
#
#         :step_mode: step_linear
#         :required variables: "initial","min","max","epsilon_step"
#         :function:
#             .. math::
#
#                 \epsilon_{t+1} = \epsilon_{t}+\frac{(max-min)}{epsilon_step}
#
#         :note: t = step.
#
#         ---------
#
#         :step_mode: episode_base
#         :required variables: "initial","min","max","alpa","episode"
#         :function:
#             .. math::
#
#                 \epsilon_{t+1} = 1 - \frac{1}{(1+ episode * alpa)}
#
#         ---------
#
#         :step_mode: constant
#         :required variables: "value"
#         :function:
#             .. math::
#
#                 \epsilon = value
#
#         ---------
#
#
#
#     """
#     def __init__(self, step_mode = "step_linear", initial = 1.0, min = 0.0,
#                     max = 1.0, epsilon_step = 25000, epsilon=0.1 , test_epsilon = 0.0, alpha=1.0):
#
#         function_list={
#         "step_linear":self._step_linear,
#         "episode_inverse":self._episode_inverse,
#         "constant":self._constant,
#         }
#
#         assert step_mode in function_list, "there is no {} as step_mode.".format(step_mode)
#
#         self.func=function_list[step_mode]
#
#
#         self.max = max
#         self.min = min
#         self.initial = initial
#         self.test_epsilon = test_epsilon
#         self.epsilon=max
#
#
#         if step_mode =="step_linear":
#             self.epsilon_step =  epsilon_step
#             self.step_size = (max-min)/epsilon_step
#
#         elif step_mode=="episode_inverse":
#             assert alpha is not None, "set alpha"
#             self.alpha = alpha
#
#         elif step_mode=="constant":
#             assert epsilon is not None, "set epsilon"
#             self.constant =  epsilon
#
#
#     def __call__(self,step,episode,epoch):
#         self.epsilon = self.func(step,episode,epoch)
#         return self.epsilon
#
#     def _step_linear(self,step, episode, epoch):
#         return self._clip(self.initial-self.step_size*step)
#
#     def _episode_inverse(self,step, episode, epoch):
#         return self._clip(self.min+(self.initial-self.min)/(1+episode*self.alpha))
#
#     def _constant(self,step, episode, epoch):
#         return np.clip(self.constant,0,1)
#
#     def _clip(self,greedy):
#         """
#         clipping function
#         """
#         greedy = np.clip(greedy, self.min, self.max)
#         return greedy
