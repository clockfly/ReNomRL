import numpy as np

class EpsilonUpdate(object):

    def __init__(self, mode="step_0", **kwargs):

        self.f_dictionary={
            "step_0":self.by_step_0,
            "episode_0":self.by_episode_0
        }

        # set "step_0" by default
        self.func=self.f_dictionary[mode] if mode in self.f_dictionary else self.f_dictionary["step_0"]


        #creating obj as instance
        self.ref=kwargs

        #setting greedy as initial constant
        self.greedy=kwargs["initial"]

    def init(self):
        self.greedy=self.ref["initial"]
        return self.greedy

    def push(self):
        return self.func()

    def by_step_0(self):
        ref=self.ref
        max=ref["max"]
        min=ref["min"]
        greedy_step=ref["greedy_step"]

        self.greedy = self._clip(self.greedy+(max-min)/greedy_step)

        return self.greedy



    def by_episode_0(self):
        ref=self.ref
        max=ref["max"]
        min=ref["min"]
        greedy_step=ref["greedy_step"]
        episode=ref["nth_episode"]

        self.greedy=self._clip(greedy+(max-min)/greedy_step*episode)

        return self.greedy



    def _clip(self,greedy):
        greedy = np.clip(greedy, self.ref["min"], self.ref["max"])
        return greedy
