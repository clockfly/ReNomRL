import pytest


@pytest.fixture(params=[False, True])
def use_gpu(request):
    """
    Gpu switch for test.
    """
    yield request.param
