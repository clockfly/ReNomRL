##########################################################################
# Defining test_ev
##########################################################################

import numpy as np


def try_except(self, func):
    def inner(*args, **kwargs):
        try:
            func(*args, **kwargs)
        finally:
            self.logger.close()
    return inner


def remove_col(self, func):
    def inner(*args, **kwargs):
        return func(*args, **kwargs)[:-2]
    return inner


class DQN(object):

    def __init__(self, logger):
        logger._key_check(["state", "reward", "terminal"])
        self.logger = logger
        self.fit = try_except(self, self.fit)

    def fit(self):
        self.logger.start(30)

        s, r = 0, 0
        for i in range(30):
            s += 1
            r += np.array([np.random.randint(-2, 3), np.random.randint(0, 6)])
            t = np.random.randint(0, 2)
            self.logger.logger(state=s, reward=r, terminal=t)
            self.logger.update(1)

        self.logger.close()


simpleLogger = None
dqn = None
