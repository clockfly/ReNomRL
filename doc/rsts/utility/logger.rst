renom_rl.utility.logger : Logger Series
=============================================================

This section shows description for logger class series.

**About Iteration**

  - For modules using a single agent and non advanced steps, 1 iteration is equivalent to 1 step.
  - For modules using advantage steps, 1 iteration is equivalent to the advantage step.
  - For modules using multiple agents, 1 agent is logged.


**About Keys**

  Keys are string values which refer to variables inside algorithms. For example, if users set "reward" as a key value, "reward" variables will be viewed (or recorded.)

View `Logger Key Table <./logger_key.html>`_ for variables available to log.

.. automodule:: renom_rl.utility.logger

    .. autodata:: AVAILABLE_KEYS
        :annotation: [dict]

    .. autoclass:: Logger
        :members: logger,logger_epoch,result,result_epoch,graph,graph_epoch,graph_custom,to_csv,from_csv

    .. autoclass:: SimpleLogger
        :exclude-members: logger
