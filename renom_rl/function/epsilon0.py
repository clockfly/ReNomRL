# import numpy as np
#
# class EpsilonUpdate(object):
#     """ EpsilonUpdate Class
#     This objects allow you to change the epsilon function. These are implemented in RL framework.
#     By specifying the greedy mode ("step_linear","episode_base" etc.), you can change the epsilon mode.
#     All functions are in the range of min and max.
#
#     Args:
#         mode (string): mode of learning.
#         **kwargs: parameters such as min, max, greedy_step, episode, alpha
#
#     Example:
#         >>> update = EpsilonUpdate(mode="step_linear",initial=min_greedy,min=min_greedy,max=max_greedy,greedy_step=greedy_step)
#         >>> greedy = update.init()
#         >>> greedy = update.push()
#
#
#     Mode:
#
#         :mode: step_linear
#         :required variables: "initial","min","max","greedy_step"
#         :function:
#             .. math::
#
#                 \epsilon_{t+1}=\epsilon_{t}+\frac{(max-min)}{greedy_step}
#
#         :note: t = step.
#
#         ---------
#
#         :mode: episode_base
#         :required variables: "initial","min","max","alpa","episode"
#         :function:
#             .. math::
#
#                 \epsilon_{t+1}= 1 - \frac{1}{(1+ episode * alpa)}
#
#         ---------
#
#
#
#
#     """
#     def __init__(self, mode="step_linear", **kwargs):
#
#         self.f_dictionary={
#             "step_linear":self._step_linear,
#             "episode_base":self._episode_base,
#         }
#
#         # set "step_0" by default
#         self.func=self.f_dictionary[mode] if mode in self.f_dictionary else self.f_dictionary["step_linear"]
#
#
#         #creating obj as instance
#         self.ref=kwargs
#
#         #setting greedy as initial constant
#         self.greedy=kwargs["initial"] if "initial" in self.ref else 0
#         self.min=kwargs["min"] if "min" in self.ref else 0
#         self.max=kwargs["max"] if "max" in self.ref else 1
#         self.greedy_step=kwargs["greedy_step"] if "greedy_step" in self.ref else 250000
#
#
#     def init(self):
#         """ Initializer
#         This initialize the epsilon. This requires initial variable.
#         """
#         self.greedy=self.ref["initial"]
#         return self.greedy
#
#     def update(self,**var):
#         """ Updater
#         This push and updates the greed value. The required variables depend on the mode of function.
#         """
#         return self.func(**var)
#
#
#
#     def _step_linear(self,**var):
#         """step linear
#         Linear Incrementing function.
#         """
#         ref=self.ref
#         max=self.max
#         min=self.min
#         greedy_step=self.greedy_step
#
#         self.greedy += (max-min)/greedy_step
#
#         self.greedy = self._clip(self.greedy)
#
#         return self.greedy
#
#
#     def _episode_base(self,**var):
#         """
#         episode based increment function.
#         """
#         ref=self.ref
#         alpha=np.clip(ref["alpha"],0,1)
#         episode=var["episode"]
#
#         self.greedy=self._clip(1-1/(1+episode*alpha))
#
#         return self.greedy
#
#
#
#     def _clip(self,greedy):
#         """
#         clipping function
#         """
#         greedy = np.clip(greedy, self.min, self.max)
#         return greedy
#
#
#     def value(self):
#         """
#         returns value
#         """
#         return self.greedy
