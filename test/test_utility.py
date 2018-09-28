#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
import renom.cuda as cuda
from renom_rl.environ import BaseEnv


skipgpu = pytest.mark.skipif(not cuda.has_cuda(), reason="cuda is not installed")


class DummyEnvDiscrete(BaseEnv):

    def __init__(self, action_shape=1, state_shape=2):
        super(DummyEnvDiscrete, self).__init__(action_shape, state_shape)

    def sample(self):
        return np.random.randint(0, *self.action_shape)

    def reset(self):
        return np.random.rand(*self.state_shape)

    def step(self, action):
        return np.random.rand(*self.state_shape), int(np.random.rand() > 0.5), bool(np.random.rand() > 0.5)


class DummyEnvContinuous(BaseEnv):

    def __init__(self, action_shape=1, state_shape=2):
        super(DummyEnvContinuous, self).__init__(action_shape, state_shape)

    def sample(self):
        return np.random.rand(*self.action_shape)

    def reset(self):
        return np.random.rand(*self.state_shape)

    def step(self, action):
        return np.random.rand(*self.state_shape), int(np.random.rand() > 0.5), bool(np.random.rand() > 0.5)
