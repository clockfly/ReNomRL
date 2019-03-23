# ReNomRL beta0.4 - Reinforcement Learning Modules

- https://www.renom.jp/index.git

### Requirements

- Linux / Ubuntu
- Python 2.7 3.6
- ReNom DL 2.6.1
- numpy 1.14.5
- tqdm 4.26.0
- Gym 0.10.5
- CV2 4.0.0.21
- Homebrew (For OSX)

### Requirements For Animation

- JSAnimation
- pyglet 1.2.4
- ipython 6.2.1
- matplotlib 2.2.3

### Installation

#### Installing ReNomRL

ReNom RL can be downloaded from the URL below.

URL: https://github.com/ReNom-dev-team/ReNomRL.git

```
git clone https://github.com/ReNom-dev-team/ReNomRL.git
cd ReNomRL
pip install -r requirements.txt
python setup.py build_ext -f -i
pip install -e .
```

#### Installing Gym

OpenAI’s Gym are frequently used in tutorials, problem solving, etc. We recommend you download it.

URL : https://github.com/openai/gym#installation


### Uninstall
Go to the directory where you install ReNomRL.
From there, do the following process.
```
pip uninstall renom_rl
rm -rf ReNomRL
```


#### License
“ReNom” is provided by GRID inc., as subscribed software.  By downloading ReNom, you are agreeing to be bound by our ReNom Subscription agreement between you and GRID inc.
To use ReNom for commercial purposes, you must first obtain a paid license. Please contact us or one of our resellers.  If you are an individual wishing to use ReNom for academic, educational and/or product evaluation purposes, you may use ReNom royalty-free.
The ReNom Subscription agreements are subject to change without notice. You agree to be bound by any such revisions. You are responsible for visiting www.renom.jp to determine the latest terms to which you are bound.
