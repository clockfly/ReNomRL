"""
Keys Available When Logging
======================================================

Here, we describe the keys that are available to log.

Common Keys
------------------

"""

# from .keys import *
from .logger_base import *
from .customs import *


_common_keys = ["state","action","reward","terminal","next_state",
                "total_step","epoch_step","max_step",
                "total_episode","epoch_episode","steps_per_episode",
                "epoch","max_epoch","loss",
                "sum_reward"]


_common_keys_epoch = ["total_episode","epoch_episode",
                         "epoch","max_epoch","test_reward"]


# dqn parameters
_dqn_keys = _common_keys + ["epsilon"]
_dqn_keys_epoch = _common_keys_epoch + ["epsilon"]

# ddqn parameters
_ddqn_keys = _common_keys + ["epsilon"]
_ddqn_keys_epoch = _common_keys_epoch + ["epsilon"]

# ddpg parameters
_ddpg_keys = _common_keys + ["epsilon","noise_value"]
_ddpg_keys_epoch = _common_keys_epoch + ["epsilon","noise_value"]

# a2c parameters
_a2c_keys = _common_keys + ["entropy","advantage","num_worker"]
_a2c_keys_epoch = _common_keys_epoch + ["entropy","advantage","num_worker"]
