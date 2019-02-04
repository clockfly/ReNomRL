"""
Common Keys
=============================================================

logger function
---------------------------------------------------

.. csv-table::
   :header: "key", "description"
   :widths: 20, 80

   "state", "State the agent is in."
   "action", "Action took in environment."
   "reward", "Reward received from environment."
   "terminal", "Terminal status for each iteration."
   "next_state", "State after action occurs."
   "total_step", "Steps agent took after ``fit`` function was executed."
   "epoch_step", "Steps agent took during 1 epoch."
   "max_step", "Maximum steps that can be taken in 1 epoch."
   "total_episode", "Episodes agent experienced after ``fit`` function was executed."
   "epoch_episode", "Episodes agent experienced during 1 epoch."
   "steps_per_episode", "Steps agent took at certain episode."
   "epoch", "Number of epochs after ``fit`` function was executed."
   "max_epoch", "Maximum epochs that can be taken in training session."
   "loss", "Value of loss function. (Usually V/Q function)"
   "sum_reward", "Total rewards of 1 episode."

logger_epoch function
---------------------------------------------------

 .. csv-table::
    :header: "key", "description"
    :widths: 20, 80

    "total_episode", "Episodes agent experienced after ``fit`` function was executed."
    "epoch_episode", "Episodes agent experienced during 1 epoch."
    "epoch", "Number of epochs after ``fit`` function was executed."
    "max_epoch", "Maximum epochs that can be taken in training session."
    "test_reward", "Total rewards of ``test`` result."


--------------------------------------------------


Keys Available for Specific Algorithm
=============================================================

DQN
---------------------------------------------------

.. csv-table::
   :header: "key", "description"
   :widths: 20, 80

   "epsilon", "Epsilon value of random action choice."


DoubleDQN
---------------------------------------------------

.. csv-table::
  :header: "key", "description"
  :widths: 20, 80

  "epsilon", "Epsilon value of random action choice."


A2C
---------------------------------------------------

.. csv-table::
  :header: "key", "description"
  :widths: 20, 80

  "entropy", "Entropy value."
  "advantage", "Advantage steps."
  "num_worker", "Number of workers used to calculate."


DDPG
---------------------------------------------------

.. csv-table::
  :header: "key", "description"
  :widths: 20, 80

  "epsilon", "Epsilon value of random action choice."
  "noise_value", "Noise value."

"""


from copy import copy

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

#: dictionary value
available_keys = {"dqn":{"logger":copy(_dqn_keys),"logger_epoch":copy(_dqn_keys_epoch)},\
                  "ddqn":{"logger":copy(_ddqn_keys),"logger_epoch":copy(_ddqn_keys_epoch)},\
                  "a2c":{"logger":copy(_a2c_keys),"logger_epoch":copy(_a2c_keys_epoch)},\
                  "ddpg":{"logger":copy(_ddpg_keys),"logger_epoch":copy(_ddpg_keys_epoch)},\
                  "doc":__doc__}
