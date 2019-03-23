Installation
============

Installing ReNomRL
---------------------------------

ReNom RL can be downloaded from the URL below.

URL: https://github.com/ReNom-dev-team/ReNomRL.git

.. code-block:: sh

   git clone https://github.com/ReNom-dev-team/ReNomRL.git
   cd ReNomRL
   pip install -r requirements.txt
   python setup.py build_ext -f -i
   pip install -e .


Installing Gym
---------------------------------

OpenAI’s Gym are frequently used in tutorials, problem solving, etc. View the following site for downloading.

URL : https://github.com/openai/gym#installation

The following installation procedure is from the link above.

.. code-block:: sh

    # install Gym
    pip install gym

    # Install libraries necessary for Gym
    # OSX:
    brew install cmake boost boost-python sdl2 swig wget
    # Ubuntu 14.04:
    apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig

    # Install all games
    pip install 'gym[atari]'

.. note::

  The above installation may not work depending on the user's environment.
  Please confirm whether gym works appropriately, such as using the following code: ::

    #Gym Render Check

    import gym
    import numpy as np

    env=gym.make('Breakout-v0')
    done=False
    _=env.reset()

    env.render()
    while done == False:
    	_,_,done,_=env.step(int(np.random.randint(0,4)))
    	env.render()



**Requirements**

-	Linux/Ubuntu
-	Python 2.7 3.6
-	ReNom DL 2.7 (or over)
- numpy 1.14.5
- tqdm 4.26.0
-	Gym 0.10.5
- CV2 4.0.0.21
-	Homebrew (For OSX)

**Requirements For Animation**

- JSAnimation
- pyglet 1.2.4
- ipython 6.2.1
- matplotlib 2.2.3
