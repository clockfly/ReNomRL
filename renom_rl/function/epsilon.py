import numpy as np

class EpsilonUpdate(object):

    def __init__(self, mode="step_linear", **kwargs):

        self.f_dictionary={
            "step_linear":self._step_linear,
            "episode_base":self._episode_base,
        }

        # set "step_0" by default
        self.func=self.f_dictionary[mode] if mode in self.f_dictionary else self.f_dictionary["step_linear"]


        #creating obj as instance
        self.ref=kwargs

        #setting greedy as initial constant
        self.greedy=kwargs["initial"]


    def init(self):
        """ Initializer
        This initialize the greedy.
        """
        self.greedy=self.ref["initial"]
        return self.greedy

    def push(self):
        """ Pusher
        This push and updates the greed value.
        """
        return self.func()

    def _step_linear(self):
        """
        Linear Incrementing function.
        """
        ref=self.ref
        max=ref["max"]
        min=ref["min"]
        greedy_step=ref["greedy_step"]

        self.greedy = self._clip(self.greedy+(max-min)/greedy_step)

        return self.greedy

    def _episode_base(self):
        """
        episode based increment function.
        """
        ref=self.ref
        alpha=np.clip(ref["alpha"],0,1)
        episode=ref["nth_episode"]

        self.greedy=self._clip(1-1/(1+episode*alpa))

        return self.greedy



    def _clip(self,greedy):
        greedy = np.clip(greedy, self.ref["min"], self.ref["max"])
        return greedy
