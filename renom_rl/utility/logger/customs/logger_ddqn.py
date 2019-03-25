import numpy as np
from ..logger_base import Logger


class DoubleDQNLogger(Logger):
    def __init__(self, log_key=None, log_key_epoch=None):
        log_key = ["sum_reward"] if log_key is None else log_key
        log_key_epoch = ["test_reward"] if log_key_epoch is None else log_key_epoch
        super(DoubleDQNLogger, self).__init__(log_key=log_key, log_key_epoch=log_key_epoch)
        self.train_sum_rewards_in_epoch = 0
        self.tick = 0
        self.train_loss = 0
        self.epoch_step = 0
        self.greedy = 0

    def logger(self, **log):

        # reset
        if not log["epoch_step"]:
            self.train_sum_rewards_in_epoch = 0
            self.train_loss = 0
            self.tick = 0

        #　defining variable
        e = log["epoch"]
        greedy = log["epsilon"]
        loss = log["loss"]
        sum_reward = log["sum_reward"]
        rewards_in_epoch = self.train_sum_rewards_in_epoch
        nth_episode = log["epoch_episode"]

        #　for epoch calc
        if log["terminal"]:
            self.train_sum_rewards_in_epoch += sum_reward
            self.tick += 1
        self.train_loss += loss
        self.epoch_step = log["epoch_step"]
        self.greedy = greedy

        # 　message
        msg = "epoch {:04d} epsilon {:.4f} loss {:5.4f} rewards in epoch {:4.3f} episode {:04d} rewards in episode {:4.3f}."\
            .format(e, greedy, loss, rewards_in_epoch, nth_episode, sum_reward)

        return msg

    def logger_epoch(self, **log):
        e = log["epoch"]
        avg_error = self.train_loss / (self.epoch_step + 1)
        summed_train_reward = self.train_sum_rewards_in_epoch
        summed_test_reward = log["test_reward"]
        # avg_train_reward = np.mean(self.train_sum_rewards_in_each_episode)
        avg_train_reward = self.train_sum_rewards_in_epoch / (self.tick + 1)
        greedy = self.greedy

        # 　message
        msg = "epoch {:03d} avg_loss:{:6.4f} total reward in epoch: [train:{:4.3f} test:{:4.3}] " + \
            "avg train reward in episode:{:4.3f} epsilon :{:4.3f}"
        msg = msg.format(e, avg_error, summed_train_reward,
                         summed_test_reward, avg_train_reward, greedy)

        return msg
