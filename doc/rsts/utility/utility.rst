renom_rl.utility
==================

Utilities (such as EpsilonGreedyFilter etc.) for the ReNomRL Modules are located in this section.
There are modules that require these utilities. There are also modules which users can use to modify algorithms, such as how epsilon increases during training.

Animation
--------------------------------------------------------------

This allows users to create animation. Mainly developed for jupyter notebook.

.. toctree::
   animation

Logger
--------------------------------------------------------------

Logger allows users to custom logging data. By modifying the logger function and implementing
it to the algorithms, users can view their own original logging data, save logged data to csv
or view graph data for immediate check.

.. toctree::
   :maxdepth: 2

   logger
   logger_key


Action Filter
--------------------------------------------------------------

Action filters are used to add noise or choose random actions. In the ``renom_rl.utility.filter``, there are various modules to design action filter.
Action Filters are used for random action decision in algorithms such as DQN, DDQN etc.
Action Noise Filters are used for random action noises in algorithms such as DDPG etc.
Epsilon are used to change epsilon value during epsilon greedy decision etc.
Noise are used to generate noise.

.. toctree::
	actionfilter
	actionnoisefilter
	epsilon
	noise
	DiscreteNodeChooser




..
   renom_rl.utility.filter
