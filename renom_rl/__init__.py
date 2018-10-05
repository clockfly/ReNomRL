#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__version__ == "0.0.0b"


class AgentBase(object):

    def __init__(self):
        self._epoch_index = []
        self._train_avg_loss_list = []
        self._train_avg_reward_list = []
        self._train_reward_list = []
        self._test_reward_list = []

    def fit(self, *args, **kwargs):
        pass

    def action(self, *args, **kwargs):
        pass

    @property
    def history(self):
        return [
            {
                "epoch": self._epoch_index[i],
                "avg_loss": self._train_avg_loss_list[i],
                "avg_train_reward": self._train_avg_reward_list[i],
                "train_reward": self._train_reward_list[i],
                "test_reward": self._test_reward_list[i]
            }
            for i in range(len(self._epoch_index))]

    def test(self, *args, **kwargs):
        pass

    def _append_history(self, epoch, avg_loss, avg_reward, train_reward, test_reward):
        self._epoch_index.append(epoch)
        self._train_avg_loss_list.append(avg_loss)
        self._train_avg_reward_list.append(avg_reward)
        self._train_reward_list.append(train_reward)
        self._test_reward_list.append(test_reward)
