from copy import copy
from .logger_base import *
from .customs import *
from .keys import available_keys

# due to sphinx, it was neccessary to import small letter variable
AVAILABLE_KEYS = available_keys
"""
**AVAIABLE_KEYS**

Allows users to view keys that are avaiable for logging.

:Keys for AVAIABLE_KEYS:

    - dqn  : dqn epoch / epoch_logger keys
    - ddqn : ddqn epoch / epoch_logger keys
    - ddpg : ddpg epoch / epoch_logger keys
    - a2c  : a2c epoch / epoch_logger keys
    - doc  : documentation of avaiable keys

Examples:
    >>> from renom_rl.utility.logger import AVAILABLE_KEYS
    >>> AVAILABLE_KEYS["dqn"]
    {'logger': ['state', 'action', 'reward', 'terminal', 'next_state',
    'total_step', 'epoch_step', 'max_step', 'total_episode',
    'epoch_episode', 'steps_per_episode', 'epoch', 'max_epoch',
    'loss', 'sum_reward', 'epsilon'],
    'logger_epoch': ['total_episode', 'epoch_episode',
    'epoch', 'max_epoch', 'test_reward', 'epsilon']}

"""
