#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
from tqdm import tqdm

__version__ = "0.3.0b"


def fit_decorator(self,func):
    def inner(*args,**kwargs):
        try:
            func(*args,**kwargs)
        finally:
            if hasattr(self,"logger"):
                if isinstance(self.logger._tqdm,tqdm):
                    self.logger.close()
    return inner

def _pass_logger():
    """
    This function is used to override _assert_logger_super()
    """
    pass

def _check_implementation(agent):
    """
    This function is used to check for attribute implementation.
    """
    if not hasattr(agent,"logger"):
        raise NotImplementedError("Implement logger atttribute")


class AgentMeta(type):
    def __call__(cls, *args, **kwargs):
        self = cls.__new__(cls, *args, **kwargs)
        cls.__init__(self, *args, **kwargs)
        self._assert_logger_super()
        _check_implementation(self)
        return self



class AgentBase(object,metaclass=AgentMeta):

    def __init__(self):
        self._epoch_index = []
        self._train_avg_loss_list = []
        self._train_avg_reward_list = []
        self._train_reward_list = []
        self._test_reward_list = []
        self.fit = fit_decorator(self,self.fit)
        self._assert_logger_super=_pass_logger

    def fit(self, *args, **kwargs):
        pass

    def action(self, *args, **kwargs):
        pass

    def _assert_logger_super(self):
        """
        this is used when creating new object
        """
        raise NotImplementedError("Need to call super('class',self).__init__(log_key,record=True)  ('class':Class Name.)")


    # @property
    # def history(self):
    #     return [
    #         {
    #             "epoch": self._epoch_index[i],
    #             "avg_loss": self._train_avg_loss_list[i],
    #             "avg_train_reward": self._train_avg_reward_list[i],
    #             "train_reward": self._train_reward_list[i],
    #             "test_reward": self._test_reward_list[i]
    #         }
    #         for i in range(len(self._epoch_index))]
    #
    # def test(self, *args, **kwargs):
    #     pass
    #
    # def _append_history(self, epoch, avg_loss, avg_reward, train_reward, test_reward):
    #     self._epoch_index.append(epoch)
    #     self._train_avg_loss_list.append(avg_loss)
    #     self._train_avg_reward_list.append(avg_reward)
    #     self._train_reward_list.append(train_reward)
    #     self._test_reward_list.append(test_reward)
