#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pytest
import numpy as np
import renom as rm
from renom.cuda import set_cuda_active

import test_utility
from renom_rl.dqn import DQN


@pytest.mark.parametrize("state_size", [
    3, (2, 2)
])
def test_init(state_size):
    model = rm.Sequential([
    ])
    dqn = DQN(model, state_size, 1)
    assert isinstance(dqn._state_size, tuple)


@test_utility.skipgpu
@pytest.mark.parametrize("state_size", [
    3, (3, 3)
])
def test_update(state_size, use_gpu):
    set_cuda_active(use_gpu)
    if isinstance(state_size, tuple):
        model = rm.Sequential([
            rm.Conv2d(input_size=(3, *state_size)),
            rm.Flatten(),
            rm.Dense(1)
        ])
    else:
        model = rm.Sequential([
            rm.Dense(1, input_size=state_size)
        ])

    dqn = DQN(model, state_size, 1)

    # Clear target network.
    for layers, target_layers in zip(dqn._network, dqn._target_network):
        if hasattr(layers, "params"):
            for p in layers.params.keys():
                array = np.random.rand(*target_layers.params[p].shape)
                target_layers.params[p] = rm.Variable(array)

    # Confirm weight params are not equal.
    for layers, target_layers in zip(dqn._network, dqn._target_network):
        if hasattr(layers, "params"):
            for p in layers.params.keys():
                p1 = layers.params[p].as_ndarray()
                p2 = target_layers.params[p].as_ndarray()
                assert not np.allclose(p1, p2)

    dqn.update()
    # Confirm weight params are equal after calling update method.
    for layers, target_layers in zip(dqn._network, dqn._target_network):
        if hasattr(layers, "params"):
            for p in layers.params.keys():
                p1 = layers.params[p].as_ndarray()
                p2 = target_layers.params[p].as_ndarray()
                assert np.allclose(p1, p2)
