import numpy as np
from copy import copy, deepcopy


def _moving_average(data, min_length, max_length):
    """
    Average moving length.
    """

    res = []
    for i in range(len(data)):
        min_i = np.maximum(i - min_length, 0)
        max_i = np.minimum(i + max_length + 1, len(data))

        avg = np.sum(data[min_i:max_i], axis=0)/(max_i-min_i)
        res.append(avg)

    return np.array(res)


def _pass_logger():
    """
    This function is used to override _assert_logger_super()
    """
    pass


def _remove_col(func):
    """
    This function is used to remove `:` for tqdm.
    tqdm ver.4.19 work.
    """
    def inner(*args, **kwargs):
        return func(*args, **kwargs)[:-2]

    return inner


def _log_decorator_iter(self, log_func):
    """
    This function will decorate logging function.
    Do not delete this function.
    """
    def _decorator(**kwargs):

        kwargs = deepcopy(kwargs)

        log_msg = log_func(**kwargs)

        if log_msg:
            self.set_description(log_msg)

        rollout = True if self._rollout else False
        # assert isinstance(kwargs["terminal"],(np.ndarray,list)), "terminal must be a list or numpy array"
        terminal_exist = kwargs["terminal"] if not rollout else np.array(kwargs["terminal"]).any()
        record_step = not self._record_episode_base
        record_episode = self._record_episode_base and terminal_exist

        if self._record and (record_step or record_episode):
            if not rollout:
                for key in self._log_dic:
                    # if key in kwargs:   maybe redundant
                    self._log_dic[key].append(kwargs[key])
            else:
                terminal_list = np.array(kwargs["terminal"]).squeeze()
                terminal_index = np.where(terminal_list)

                for key in self._log_dic:
                    # if key in kwargs:  maybe redundant
                    key_value_list = np.array(kwargs[key])
                    for value in key_value_list[terminal_index]:
                        self._log_dic[key].append(value)

    return _decorator


def _log_decorator_epoch(self, log_func):
    """
    This function will decorate logging function for epoch run.
    Do not delete this function.
    """
    def _decorator2(**kwargs):

        kwargs = deepcopy(kwargs)

        log_msg = log_func(**kwargs)

        if log_msg:
            self.set_description(log_msg)

        if self._record:
            for key in self._log_dic_epoch:
                if key in kwargs:
                    self._log_dic_epoch[key].append(kwargs[key])

    return _decorator2


def _to_csv_data(log_dic):
    """
    Creates data structure to write in csv.
    """
    key_list = log_dic.keys()
    header = {val: val for val in key_list}
    first_key = next(iter(log_dic))
    data_length = len(log_dic[first_key])

    row_data = []

    for i in range(data_length):
        row_data.append({key_i:
                         log_dic[key_i][i].tolist() if isinstance(log_dic[key_i][i], np.ndarray)
                         else log_dic[key_i][i]
                         for key_i in log_dic})

    return header, row_data
