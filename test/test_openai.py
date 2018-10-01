import pytest
from renom_rl.environ.openai import Breakout, Pendulum


@pytest.mark.parametrize("env", [
    Breakout,
    Pendulum
])
def test_env(env):
    e = env()
