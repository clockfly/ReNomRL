import pytest
import matplotlib.pyplot as plt
import glob2
import os


def not_show():
    plt.close()


@pytest.fixture(scope='session', autouse=True)
def scope_session():

    # setting
    plt.show = not_show

    print("start")
    yield
    print("teardown after session")


pytest.fixture(scope='module', autouse=True)


def scope_module():

    print("module start")
    yield
    print("module done")


def pytest_namespace():
    return {"logger": 1}
