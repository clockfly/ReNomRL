import numpy as np
from ..logger_base import Logger


class A2CLoggerD(Logger):
    def __init__(self, log_key=None, log_key_epoch=None):
        log_key = ["sum_reward"] if log_key is None else log_key
        log_key_epoch = ["test_reward"] if log_key_epoch is None else log_key_epoch
        super(A2CLoggerD, self).__init__(log_key=log_key, log_key_epoch=log_key_epoch)
        self.train_sum_rewards_in_epoch = 0
        self.tick = 0
        self.sum_reward_per_epoch = 0
        self.train_loss = 0
        self.epoch_step = 0
        self.nth_episode = 0
        self.sum_reward = 0

    def logger(self, **log):

        # reset
        if not log["epoch_step"][0]:
            self.train_sum_rewards_in_epoch = 0
            self.train_loss = 0
            self.sum_reward_per_epoch = 0
            self.tick = 0
            self.terminal = []
            self.max_reward = 0

        #　defining variable
        e = log["epoch"][-1]
        loss = log["loss"][-1]
        terminal_exist = log["terminal"].any()
        terminal_index = np.where(log["terminal"].squeeze())
        nth_episode = self.nth_episode
        sum_reward = self.sum_reward
        total = self.train_sum_rewards_in_epoch

        #　for epoch calc
        if log["terminal"].any():
            self.max_reward = np.max(log["reward"])
            self.train_sum_rewards_in_epoch += np.sum(log["sum_reward"][terminal_index])
            self.tick += len(log["sum_reward"][terminal_index])
            self.nth_episode = log["epoch_episode"][terminal_index][-1]
            self.sum_reward = log["sum_reward"][terminal_index][-1]

        self.train_loss += loss
        self.epoch_step = log["epoch_step"][-1]
        self.sum_reward_per_epoch += np.sum(log["reward"])

        msg = "agent[0], epoch {:04d} loss {:5.4f} rewards in epoch {:4.3f}  episode {:04.1f} rewards in episode {:4.3f}."\
            .format(e, loss, total, nth_episode, sum_reward)

        return msg

    def logger_epoch(self, **log):
        e = log["epoch"]
        avg_loss = self.train_loss / self.epoch_step + 1
        avg_train_reward = self.train_sum_rewards_in_epoch/(self.tick + 1)
        train_reward = self.sum_reward_per_epoch
        test_reward = log["test_reward"]

        msg = "epoch {:03d} avg_loss:{:6.4f} total reward in epoch: [train:{:4.3f} test:{:4.3}] " + \
            "avg train reward per episode:{:4.3f}"
        msg = msg.format(e, avg_loss, train_reward,
                         test_reward, avg_train_reward)

        return msg
