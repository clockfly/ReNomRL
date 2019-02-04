How to Use
============================

Overview
------------------------------------------------

When using neural networks in reinforcement learning, neural network is used as an agent with multiple signals as input and action as output. However, due to the difference in problems users are facing( such as what information the agent acquires from the environment, or what types of actions are required ) we should not only define the agent structure, but also the environment as well.

ReNom RL has multiple built-in algorithm, such as DQN, A3C etc. When implementing reinforcement learning with ReNom RL, the following 3 actions are required:

1. Environment Preparation
2. Model Preparation
3. Implementation of Reinforcement Learning

1-Environment Preparation
------------------------------------------------

In order to use quickly apply the environment, fitting the environment structure according to BaseEnv module is required. In this section, we will introduce 2 ways of preparing the environment: using pre-prepared environment and implementing environment from scratch.

Using Pre-prepared Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We prepared environment models that uses Open AI. For example, if the user wants to use breakout model for its test, we could call the environment as shown below.

.. code-block:: python

    from renom_rl.environ.openai import Breakout
    env = Breakout()

Implementing Environment from Scratch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When creating an original environment, the object must be inherited, overwriting the variables and the function as mentioned below:

* **action_shape:** the shape of action
* **state_shape:** the shape of state
* **reset():** the function that resets the environment
* **sample():** the function that chooses random action
* **step():** the function that outputs state, reward, terminal when taking a step

For example, when creating an original environment called CustomEnv(), the implemetation can be done as shown below:

.. code-block:: python

  class CustomEnv(BaseEnv):

      def __init__(self, env):
          self.action_shape = (2,)
          self.state_shape = (4,)

          self.env=env
          self.step_continue=0
          self.reward=0



      def reset(self):
          return self.env.reset()


      def sample(self):
          rand=env.action_space.sample()
          return rand

      def step(self, action):
          state,_,terminal,_=env.step(int(action))

          self.step_continue+=1
          reward=0

          if terminal:
              if self.step_continue >= 200:
                  reward=1
              else:
                  reward=-1

          self.reward=reward

          return state, reward, terminal

    new_env=CustomEnv()


2-Model Preparation
------------------------------------------------

In this section, we use ReNom DL to build a model. Define the model as shown below when using a standard neural network.

.. code-block:: python

  import renom as rm
  q_network = rm.Sequential([rm.Dense(30, ignore_bias=True),
                             rm.Relu(),
                             rm.Dense(30, ignore_bias=True),
                             rm.Relu(),
                             rm.Dense(2, ignore_bias=True)])

3-Implementation of Reinforcement Learning
---------------------------------------------------------------------

After preparing the environment and the model, we now implement using a certain algorithm. The script below describes the algorithm for DQN.

.. code-block:: python

    from renom_rl.discrete.dqn import DQN

    model = DQN(custom_env, q_network)

After finishing the model, we run the module by implementing as shown below:

.. code-block:: python

    result = model.fit()

By implement as shown above, we can run DQN. For more information, please refer the API page on environment, and other algorithms.
