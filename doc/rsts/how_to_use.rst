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

    import gym
    from renom_rl.environ import BaseEnv

    env = gym.make('BreakoutNoFrameskip-v4')

    class CustomEnv(BaseEnv):

        def __init__(self, env):
            self.env = env
            self.action_shape = 4
            self.state_shape = (4, 84, 84)
            self.previous_frames = []
            self._reset_flag = True
            self._last_live = 5
            super(CustomEnv, self).__init__()

        def reset(self):
            if self._reset_flag:
                self._reset_flag = False
                self.env.reset()
            n_step = np.random.randint(4, 32+1)
            for _ in range(n_step):
                state, _, _ = self.step(self.env.action_space.sample())
            return state

        def sample(self):
            return self.env.action_space.sample()

        def render(self):
            self.env.render()

        def _preprocess(self, state):
            resized_image = Image.fromarray(state).resize((84, 110)).convert('L')
            image_array = np.asarray(resized_image)/255.
            final_image = image_array[26:110]
            # Confirm that the image is processed correctly.
            # Image.fromarray(np.clip(final_image.reshape(84, 84)*255, 0, 255).astype(np.uint8)).save("test.png")
            return final_image

        def step(self, action):
            state_list = []
            reward_list = []
            terminal = False
            for _ in range(4):
                # Use last frame. Other frames will be skipped.
                s, r, t, info = self.env.step(action)
                state = self._preprocess(s)
                reward_list.append(r)
                if self._last_live > info["ale.lives"]:
                    t = True
                    self._last_live = info["ale.lives"]
                    if self._last_live > 0:
                        self._reset_flag = False
                    else:
                        self._last_live = 5
                        self._reset_flag = True
                if t:
                    terminal = True

            if len(self.previous_frames) > 3:
                self.previous_frames = self.previous_frames[1:] + [state]
            else:
                self.previous_frames += [state]
            state = np.stack(self.previous_frames)
            return state, np.array(np.sum(reward_list) > 0), terminal

    new_env=CustomEnv()


2-Model Preparation
------------------------------------------------

In this section, we use ReNom DL to build a model. Define the model as shown below when using a standard neural network.

.. code-block:: python

    q_network = rm.Sequential([rm.Conv2d(32, filter=8, stride=4),
                               rm.Relu(),
                               rm.Conv2d(64, filter=4, stride=2),
                               rm.Relu(),
                               rm.Conv2d(64, filter=3, stride=1),
                               rm.Relu(),
                               rm.Flatten(),
                               rm.Dense(512),
                               rm.Relu(),
                               rm.Dense(custom_env.action_shape)])

3-Implementation of Reinforcement Learning
---------------------------------------------------------------------

After preparing the environment and the model, we now implement using a certain algorithm. The script below describes the algorithm for DQN.

.. code-block:: python

    import renom as rm
    from renom_rl.discrete.dqn import DQN

    model = DQN(custom_env, q_network)

After finishing the model, we run the module by implementing as shown below:

.. code-block:: python

    result = model.fit(render=False, greedy_step=1000000, random_step=5000, update_period=10000)

By implement as shown above, we can run DQN. For more information, please refer the API page on environment, and other algorithms.
