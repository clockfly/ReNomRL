import numpy as np
from ..logger_base import Logger


class DDPGLogger(Logger):
    def __init__(self, log_key=None, log_key_epoch=None):
        log_key = ["sum_reward"] if log_key is None else log_key
        log_key_epoch = ["test_reward"] if log_key_epoch is None else log_key_epoch
        super(DDPGLogger, self).__init__(log_key=log_key, log_key_epoch=log_key_epoch)
        self.train_sum_rewards_in_epoch = 0
        self.train_loss = 0
        self.sum_reward_per_epoch = 0
        self.tick = 0
        self.e_rate = 0
        self.epoch_step = 0

    def logger(self, **log):

        # reset
        if not log["epoch_step"]:
            self.train_sum_rewards_in_epoch = 0
            self.train_loss = 0
            self.sum_reward_per_epoch = 0
            self.tick = 0

        #　defining variable
        e = log["epoch"]
        reward = log["reward"]
        total_reward = self.sum_reward_per_epoch + reward

        #　for epoch calc
        if log["terminal"]:
            self.train_sum_rewards_in_epoch += log["sum_reward"]
            self.tick += 1
        self.train_loss += log["loss"]
        self.epoch_step = log["epoch_step"]
        self.e_rate = log["epsilon"]
        self.sum_reward_per_epoch += reward

        return "epoch: {:03d} Each step reward:{:0.2f}".format(e, total_reward)

    def logger_epoch(self, **log):
        e = log["epoch"]
        avg_loss = self.train_loss / self.epoch_step + 1
        avg_train_reward = self.train_sum_rewards_in_epoch/(self.tick + 1)
        train_reward = self.sum_reward_per_epoch
        test_reward = log["test_reward"]
        e_rate = self.e_rate

        msg = "epoch {:03d} avg_loss:{:6.3f} total_reward [train:{:5.3f} test:{:5.3f}] avg train reward in episode:{:4.3f} e-rate:{:5.3f}"
        msg = msg.format(e, avg_loss, train_reward, test_reward, avg_train_reward, e_rate)

        return msg
