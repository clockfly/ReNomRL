#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
import renom.cuda as cuda
skipgpu = pytest.mark.skipif(not cuda.has_cuda(), reason="cuda is not installed")
