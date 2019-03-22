import numpy as np
import pytest
from functools import partial
from test_environ import DQN
# from copy import copy
import os
from renom_rl.utility.logger import Logger, SimpleLogger

##########################################################################

##########################################################################


class FunctionalLogger(Logger):
    def __init__(self, **kwargs):
        # if "msg" in kwargs:
        #     del kwargs["msg"]
        super(FunctionalLogger, self).__init__(**kwargs)
        self.reward_previous = 0
        self.reward = 0
        self.total_list = []
        self.state = 0
        self.total = 0

    def logger(self, **log):
        self.state = log["state"]
        self.reward = log["reward"]

        flag = np.sum(self.reward - self.reward_previous)*np.random.randint(-1, 2)

        self.total += flag if flag >= 0 else 0

        if flag:
            self.total_list.append(self.reward)

        self.reward_previous = self.reward

        if flag:
            return "state----{}/reward---{}/total----{}".format(self.reward, self.state, flag)
        else:
            return "minus"


# @pytest.fixture()
# def logger_parameter():
#     return "FunctionalLogger"


@pytest.fixture(scope="class", params=[SimpleLogger, FunctionalLogger])
def logger(request):
    yield request.param
    dirname = os.path.dirname(os.path.abspath(__file__))
    test_csv_path = dirname+"/test_csv.csv"
    if os.path.exists(test_csv_path):
        os.remove(test_csv_path)


@pytest.fixture(scope="class", params=[
    [{"log_key": ["state", "reward"], "msg":"This is {} reward:{}", "record_episode_base":True},
     {"log_key": ["state", "reward"], "msg":"This is {} reward:{}", "show_bar":False},
     {"log_key": ["state", "reward"], "msg":"This is {} reward:{}", "disable":True},
     {"log_key": ["stat", "reward"], "msg":"This is {} reward:{}"}]
])
def all_param(request):
    yield request.param


@pytest.mark.usefixtures("all_param", "logger")
class Test_SimpleLogger(object):

    def delete_msg(self, logger, kwargs):
        if logger.__name__ == "FunctionalLogger" and "msg" in kwargs:
            del kwargs["msg"]
        return kwargs

    @pytest.fixture()
    def Logger(self, logger):
        return logger

    @pytest.fixture()
    def key_normal(self, Logger, all_param):

        return self.delete_msg(Logger, all_param[0])

    @pytest.fixture()
    def key_bar_hide(self, Logger, all_param):
        return self.delete_msg(Logger, all_param[1])

    @pytest.fixture()
    def key_disable(self, Logger, all_param):
        return self.delete_msg(Logger, all_param[2])

    @pytest.fixture()
    def key_bad_log_key(self, Logger, all_param):
        return self.delete_msg(Logger, all_param[3])

    def test_instance(self, Logger, key_normal):
        pytest.logger = Logger(**key_normal)

    def test_logging(self):
        dqn = DQN(pytest.logger)
        dqn.fit()
        pytest.dqn = dqn

    def test_graph(self):
        pytest.logger.graph(x_key="state", y_key="reward",
                            figsize=(6, 6), dpi=100, average_range=1)

    def test_to_csv(self):
        pytest.logger.to_csv("./test_csv.csv", overwrite=True)

    def test_reset(self):
        pytest.logger.reset()
        res_dic = pytest.logger.result()
        for k in res_dic:
            assert res_dic[k] == [], "data was not empty"

    def test_from_csv(self):
        pytest.logger.from_csv("./test_csv.csv")
        pytest.logger.graph(x_key="state", y_key="reward", dpi=10, grid=True)

    def test_show_tqdm_bar(self, Logger, key_bar_hide):
        pytest.testlogger = Logger(**key_bar_hide)
        pytest.dqn = DQN(pytest.logger)
        pytest.dqn.fit()

    def test_disable_tqdm(self, Logger, key_disable):
        pytest.logger = Logger(**key_disable)
        pytest.dqn = DQN(pytest.logger)
        pytest.dqn.fit()

    def test_log_key(self, Logger, key_bad_log_key):
        with pytest.raises(Exception) as excinfo:
            pytest.logger = Logger(**key_bad_log_key)
            pytest.dqn = DQN(pytest.logger)
            pytest.dqn.fit()

        assert "does not exist as logging key in this module" in str(excinfo.value)

    def test_try_error(self):
        logger = SimpleLogger(log_key=["state", "reward"], msg="This is {} reward:{}")
        dqn = DQN(logger)
        dqn.fit()
