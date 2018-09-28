#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

import renom as rm
import renom.cuda as cuda
from renom_rl.environ import BaseEnv
from renom_rl.discrete.dqn import DQN
from renom_rl.discrete.double_dqn import DoubleDQN

from test_utility import DummyEnvDiscrete

@pytest.mark.parametrize("agent, environ, fit_args", [
    [DQN, DummyEnvDiscrete, {"train_frequency":1}],
    [DoubleDQN, DummyEnvDiscrete, {}],
])
def test_fit(agent, environ, fit_args, use_gpu):
    cuda.set_cuda_active(use_gpu)
    action_shape = (1,)
    state_shape = (2,)
    env = environ(action_shape, state_shape)
    network = rm.Sequential([
        rm.Dense(5),
        rm.Dense(action_shape[0]),
    ])
    model = agent(env, network)
    action = model.action(np.random.rand(*state_shape))
    assert action.shape == action_shape

    ## Check fit
    model.fit(epoch=1, epoch_step=10, batch_size=4, random_step=20, test_step=10, **fit_args)
    print(model.history)


