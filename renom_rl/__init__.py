#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
from tqdm import tqdm

__version__ = "0.4.0b"


def fit_decorator(self, func):
    def inner(*args, **kwargs):
        try:
            func(*args, **kwargs)
        finally:
            if hasattr(self, "logger"):
                if isinstance(self.logger._tqdm, tqdm):
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
    if not hasattr(agent, "logger"):
        raise NotImplementedError("Implement logger atttribute")


class AgentMeta(type):
    def __call__(cls, *args, **kwargs):
        self = cls.__new__(cls, *args, **kwargs)
        cls.__init__(self, *args, **kwargs)
        self._assert_logger_super()
        _check_implementation(self)
        return self


class AgentBase(object, metaclass=AgentMeta):

    def __init__(self):
        self._epoch_index = []
        self._train_avg_loss_list = []
        self._train_avg_reward_list = []
        self._train_reward_list = []
        self._test_reward_list = []
        self.fit = fit_decorator(self, self.fit)
        self._assert_logger_super = _pass_logger

    def fit(self, *args, **kwargs):
        pass

    def action(self, *args, **kwargs):
        pass

    def _assert_logger_super(self):
        """
        this is used when creating new object
        """
        raise NotImplementedError(
            "Need to call super('class',self).__init__(log_key,record=True)  ('class':Class Name.)")
