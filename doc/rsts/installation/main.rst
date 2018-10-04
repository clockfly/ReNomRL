Installation
============

ReNom RL can be downloaded from the URL below.

URL: https://github.com/ReNom-dev-team/ReNomRL.git

.. code-block:: sh

   git clone https://github.com/ReNom-dev-team/ReNomRL.git
   cd ReNomRL
   python setup.py build_ext -f -i
   pip install -e .

OpenAI’s Gym are frequently used in tutorials, problem solving, etc. We recommend you download it.
    
URL : https://github.com/openai/gym#installation

.. code-block:: sh

    # install Gym
    pip install gym

    # Install libraries necessary for Gym
    # OSX:
    brew install cmake boost boost-python sdl2 swig wget
    # Ubuntu 14.04:
    apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
    
    # Install all games
     pip install 'gym[all]'


**Requirements**   

-	Python 2.7 3.6
-	ReNom DL
-	OpenAI Gym
-	Homebrew (For OSX)
