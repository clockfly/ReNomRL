import numpy as np
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
from copy import copy, deepcopy
from .functions import _moving_average, _pass_logger, _remove_col, _log_decorator_iter, _log_decorator_epoch, _to_csv_data


class LoggerMeta(type):
    def __call__(cls, *args, **kwargs):
        self = cls.__new__(cls, *args, **kwargs)
        cls.__init__(self, *args, **kwargs)
        self._assert_logger_super()
        return self


class Logger(object, metaclass=LoggerMeta):
    """
    **Logger Module**

    This class logs various data of each module.\n
    By setting ``log_key`` , ``log_key_epoch`` , this class will record data based on keys for every iteration.
    Keys specified for these argument are mainly used for plotting graph and csv output.
    ``log_key`` , ``log_key_epoch`` argument must be a list of strings which exist in the algorithm. \n
    ``logger(**log)`` function reads all keys that are available to read at every iteration and returns as a progress bar message. Simultaneously, data specified at ``log_key`` will be memorized as iteration data. **(Overriding Required)**
    ``logger_epoch(**log)`` function reads all keys that are available to read at every end of epoch and returns as a progress bar message. Simultaneously, data specified at ``log_key_epoch`` will be memorized as epoch data. **(Not Mandatory)**\n
    Users must also call super class ``super().__init__`` when initializing.\n


    Args:

        log_key(list): List of logging keys for each iteration.
        log_key_epoch(list): List of logging keys for each epoch.
        record(boolean): Keeps data for graph and csv. Default is True.
        record_episode_base(boolean): Keeps data when ``record`` is True and episode changes. Default is True.
        show_bar(boolean): Shows bar. Default is True.
        disable(boolean): Disables tqdm. Default is False.


    Examples:
        >>> import numpy as np
        >>> class Original(Logger):
        ...     def __init__(self,log_key):
        ...         super(Original,self).__init__(log_key,record_episode_base=False)
        ...         self.reward_previous = 0
        ...         self.reward = 0
        ...         self.total_list = []
        ...         self.state = 0
        ...         self.total = 0
        ...
        ...     def logger(self,**log):
        ...         self.state = log["state"]
        ...         self.reward = log["reward"]
        ...         self.total += log["reward"]
        ...
        ...         return "state----{}/reward---{}/total----{}".format(self.state, self.reward, self.total)
        ...
        >>>
        >>>
        >>> import renom as rm
        >>>
        >>> from renom_rl.environ.openai import CartPole00
        >>> from renom_rl.discrete.dqn import DQN
        >>>
        >>> network = rm.Sequential([rm.Dense(32),rm.Relu(),rm.Dense(32),rm.Relu(),rm.Dense(2)])
        >>>
        >>> logger = Original(["reward"])
        >>>
        >>> dqn=DQN(env=CartPole00(),q_network=network,logger=logger)
        state----[-0.00528582  0.76312646 -0.00763515 -1.1157825 ]/reward---0/total-----39: 100%|██████████████████████████████████████| 500/500 [00:01<00:00, 438.39it/s]


    .. note::

        Note that keys ``log_key`` does not suppress logger from reading values. For example, users can specify ``log_key`` as: ::

            log_k = ["state","reward"]

            class Original(Logger):
                 def __init__(self,log_key):
                     super(Original,self).__init__(log_key)
                     .....

            original = Original(log_key = log_k)


        and still view other keys such as "next_state", "terminal" at ``logger(**log)`` function, as shown below: ::

            def logger(**log):
                log_key = log["next_state"]
                terminal = log["terminal"]

                return .....

        However, users cannot graph data for "next_state", "terminal" at ``graph`` function etc..
        Users will only be able to graph using key "state", "reward" at ``graph`` function etc. ::

            graph(y_key="reward")  # Pass
            graph(y_key="terminal")  # Error

    """

    def __init__(self, log_key=None, log_key_epoch=None, record=True, record_episode_base=True, show_bar=True, disable=False):

        log_key = [] if log_key is None else log_key
        assert isinstance(log_key, list), "log_key must be a list."

        log_key_epoch = [] if log_key_epoch is None else log_key_epoch
        assert isinstance(log_key_epoch, list), "log_key_epoch must be a list."

        log_dic = {}
        for key in log_key:
            log_dic[key] = []

        log_dic_epoch = {}
        for key in log_key_epoch:
            log_dic_epoch[key] = []

        self._log_dic = log_dic
        self._log_dic_epoch = log_dic_epoch
        self._record = record
        self._rollout = False
        self._record_episode_base = record_episode_base
        self._assert_logger_super = _pass_logger
        self.logger = _log_decorator_iter(self, self.logger)
        self.logger_epoch = _log_decorator_epoch(self, self.logger_epoch)
        self._show_bar = show_bar
        self._tqdm = None
        self._disable = disable

    def start(self, length):
        """
        Initializes tqdm.

        Args:

            length(float): length of tqdm.

        """
        assert length, "set argument `length`"
        if not self._show_bar:
            self._tqdm = tqdm(range(1), total=1, bar_format="-{desc}", disable=self._disable)
            self._tqdm.__repr__ = _remove_col(self._tqdm.__repr__)
        else:
            self._tqdm = tqdm(range(length), disable=self._disable)

    def _key_check(self, log_key=[], log_key_epoch=[]):
        """
        Checks key. This will be called from each algorithm.
        """
        for key in self._log_dic:
            assert key in log_key, "{} does not exist as logging key in this module. Reset log_key.".format(
                key)

        for key in self._log_dic_epoch:
            assert key in log_key_epoch, "{} does not exist as logging key in this module. Reset log_key.".format(
                key)

    def reset(self):
        """
        Resets logged data.

        Examples:
            >>> logger.reset()
            >>> logger.result()
            {'reward': []}
        """
        for key in self._log_dic:
            self._log_dic[key] = []

    def logger(self, **log):
        """
        This function will be called for every iteration.
        **Override this function when creating custom logger**

        Args:
            log(dictionary): Data input from every iteration. Keys are logging keys.

        Returns:
            (str): Message required for progress view.

        Examples:

            >>> class Some_Logger(Logger):
            ...     def __init__(self,log_key):
            ...            ...
            ...
            ...     def logger(self,**log):
            ...         self.state = log["state"]
            ...         self.reward = log["reward"]
            ...         self.total += log["reward"]
            ...
            ...         return "making original reward:{}".format(self.reward)
            ...
            >>>
            >>> dqn=DQN(env=... , q_network= ... ,logger=Some_Logger)
            making original reward:0: 50%|████████████████████                  | 250/500 [00:01<00:00, 438.39it/s]

        """

        raise NotImplementedError("Please override `logger` method.")

    def logger_epoch(self, **log):
        """
        This function will be called when 1 epoch is done. Due to its similiarity, view ``logger`` function for detail.
        **Override this function when creating custom logger**
        """
        pass

    def logger_episode(self, **log):
        """
        This function will be called when 1 episode is done. Due to its similiarity, view ``logger`` function for detail.
        **Override this function when creating custom logger**
        """
        pass

    def result(self, *args):
        """
        Returns dictionary of data that were specified as log_key.
        If argument is blank, then all output will be shown.

        Args:
            *args(string): Strings of arguments.

        Returns:
            (dict): Dictionary of logging data.

        Examples:
            >>> logger.result()
            {'reward': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, ....]}
            >>>
            >>> logger.result("reward")
            {'reward': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, ....]}
            >>>
            >>> logger.result("reward","state")
            {'reward': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, ....],
            'state': [....],}

        """
        if args:
            res = {}
            for key in args:
                assert key in self._log_dic, "no such key as {} in log_key".format(key)
                res[key] = self._log_dic[key]
            return res
        else:
            return self._log_dic

    def result_epoch(self, *args):
        """
        Returns dictionary of data that were specified as log_key_epoch.
        If argument is blank, then all output will be shown.

        """
        if args:
            res = {}
            for key in args:
                assert key in self._log_dic_epoch, "no such key as {} in log_key_epoch".format(key)
                res[key] = self._log_dic_epoch[key]
            return res
        else:
            return self._log_dic_epoch

    def graph(self, y_key, x_key=None, x_lim=None, y_lim=None,
              x_interval=0, y_interval=0, figsize=None,
              dpi=100, average_range=0, plot_label=None, legend=None, grid=True):
        """
        Shows plot from recorded data. Keys must be from ``log_key`` . if ``x_key`` is None,
        ``y_key`` will be plot based on its length. Note that this function is focused on
        quick view so if detailed view is required, save data as csv.(refer ``to_csv`` function)

        Args:
            y_key (string): Key for Y (vertical) axis. 2-D data is allowed.
            x_key (string): Key for X (horizontal) axis. This must be 1-D data. Default is None.
            x_lim (list): [min,max] range for X axis. Default is min,max of x_key data.
            y_lim (list): [min,max] range for Y axis. Default is min,max of y_key data.
            x_interval (float): Interval of ticks for Y axis. Default is None (ticks: auto).
            y_interval(float): Interval of ticks for X axis. Default is None (ticks: auto).
            figsize (tuple): When (a,b) is input, plot increase to a*x_axis, b*y_axis.
            dpi (int): Digital Pixel Image. Default is 100.
            average_length (int or list): Creates average plot in [i - min, i + max] range \
                                    (i being plotted point), when [min,max] is set. If int \
                                    type is set, it becomes [average_length,average_length].\
                                     Default is 0.
            plot_label (string): Names label of plot when legend appears. Default is None.
            legend (bool or dictionary): Shows Legend. If dictionary, legend's property will be set based on its value. Default is None (False).
            grid (boolean): Shows grid based on ticks. Default is True.

        Examples:
            >>> logger.graph(y_key="reward",figsize=(6,6),dpi=100,average_range=[1,1])

        """

        # x_data, y_data
        assert y_key in self._log_dic,\
            "set axis to keys set in log_key variable. Avaiable variables are {}".format(
                list(self._log_dic.keys()))

        y_data = np.array(self._log_dic[y_key])

        if x_key is not None:
            assert x_key in self._log_dic,\
                "set axis to keys set in log_key variable. Avaiable variables are {}".format(
                    list(self._log_dic.keys()))
            x_data = np.array(self._log_dic[x_key])

        else:
            x_data = None
            x_key = "episodes" if self._record_episode_base else "steps"

        # creating custom graph
        self.graph_custom(y_data=y_data, x_data=x_data, y_label=y_key, x_label=x_key, x_lim=x_lim, y_lim=y_lim,
                          x_interval=x_interval, y_interval=y_interval, figsize=figsize,
                          dpi=dpi, average_range=average_range, plot_label=plot_label, legend=legend, grid=grid)

    def graph_epoch(self, y_key, x_key=None, x_lim=None, y_lim=None,
                    x_interval=0, y_interval=0, figsize=None,
                    dpi=100, average_range=0, plot_label=None, legend=None, grid=True):
        """
        Shows plot from recorded data at every epoch. View the function above for details.
        """

        # x_data, y_data
        assert y_key in self._log_dic_epoch,\
            "set axis to keys set in log_key variable. Avaiable variables are {}".format(
                list(self._log_dic_epoch.keys()))

        y_data = np.array(self._log_dic_epoch[y_key])

        if x_key is not None:
            assert x_key in self._log_dic_epoch,\
                "set axis to keys set in log_key variable. Avaiable variables are {}".format(
                    list(self._log_dic_epoch.keys()))
            x_data = np.array(self._log_dic_epoch[x_key])
        else:
            x_data = None
            x_key = "epoch"

        # creating custom graph
        self.graph_custom(y_data=y_data, x_data=x_data, y_label=y_key, x_label=x_key, x_lim=x_lim, y_lim=y_lim,
                          x_interval=x_interval, y_interval=y_interval, figsize=figsize,
                          dpi=dpi, average_range=average_range, plot_label=plot_label, legend=legend, grid=grid)

    # for custom graph
    @classmethod
    def graph_custom(cls, y_data, x_data=None, y_label="", x_label="", x_lim=None, y_lim=None,
                     x_interval=0, y_interval=0, figsize=None,
                     dpi=100, average_range=0, plot_label=None, legend=None, grid=True):
        """
        This function allows users to quickly create graph when custom creating own data.\n
        refer ``graph`` for other arguments.

        Args:

            y_data (numpy): Y (vertical) axis data. 2-D data is allowed.
            x_data (numpy): X (horizontal) axis data. This must be 1-D data. Default is None.
            y_label (string):  Y (vertical) axis label.
            x_label (string):  X (vertical) axis label.

        Examples:
            >>> # suppose logger.total has a 2D list
            >>> array_list=np.array(logger.total_list)[:,1]
            >>> logger.graph_custom(array_list,y_label="this is y",x_label="this is x",x_interval=5)

        """
        if figsize:
            plt.figure(figsize=figsize, dpi=dpi)
        else:
            plt.figure()

        cls.graph_attribute(plt, y_data=y_data, x_data=x_data, y_label=y_label, x_label=x_label, x_lim=x_lim, y_lim=y_lim,
                            x_interval=x_interval, y_interval=y_interval,
                            dpi=dpi, average_range=average_range, plot_label=plot_label, legend=legend, grid=grid)

        plt.grid(grid)
        plt.show()

    # for graph attributes
    @classmethod
    def graph_attribute(cls, plt_sub, y_data, x_data=None, y_label="", x_label="", x_lim=None, y_lim=None,
                        x_interval=0, y_interval=0,
                        dpi=100, average_range=0, plot_label=None, legend=None, grid=True):
        """
        This function allows users to generate graph properties more easily.\n
        refer ``graph`` for other arguments.

        Args:

            plt (matplotlib): plt object.

        Examples:
            >>> import numpy as np
            >>> from renom_rl.utility.logger import Logger
            >>> import matplotlib.pyplot as plt
            >>>
            >>> data_list={}
            >>> data_list["param1"]=np.random.random((100,2))
            >>> data_list["param2"]=np.random.random((100,2))
            >>> data_list["param3"]=np.random.random((100,2))
            >>>
            >>> plt.figure(figsize=(10,10))
            >>> for i , k in enumerate(data_list,1):
            ...     plt.subplot(len(data_list),1,i)
            ...     Logger.graph_attribute(plt,data_list[k],plot_label=["plt_a", "plt_b"],y_label=k,legend={"loc":"upper right"})
            >>>
            >>> plt.show()

        """

        y_data = np.array(y_data)
        x_data = x_data if x_data is not None else np.arange(len(y_data)) + 1

        assert len(np.shape(x_data)) <= 1 and len(np.shape(y_data)) <= 2,\
            "key dimension conditions are x_data <= 1 and y_data <= 2 when plotting"

        if x_lim:
            assert isinstance(x_lim, list) and len(x_lim) == 2,\
                "x_lim must be a [min,max] structure"

        if y_lim:
            assert isinstance(y_lim, list) and len(y_lim) == 2,\
                "y_lim must be a [min,max] structure"

        x_lim = x_lim if x_lim else [np.min(x_data), np.max(x_data)]
        y_lim = y_lim if y_lim else [np.min(y_data) - 0.1, np.max(y_data) + 0.1]

        plt_sub.xlim(x_lim)
        plt_sub.ylim(y_lim)

        if len(np.shape(y_data)) == 1:

            if plot_label:
                assert isinstance(plot_label, str), "plot label must be a string"
                labeling_format = plot_label
            else:
                labeling_format = "result"

            plt_sub.plot(x_data, y_data, "b", label=labeling_format)

        if len(np.shape(y_data)) == 2:

            if plot_label:
                if isinstance(plot_label, (list, tuple)):
                    labeling_format = ["{}".format(x) for x in plot_label]
                elif isinstance(plot_label, str):
                    labeling_format = ["{}[{}]".format(plot_label, i)
                                       for i in range(len(y_data[0]))]
                else:
                    raise ValueError("Must Implement tuple, list, or string for plot_label")
            else:
                if y_label:
                    labeling_format = ["{}[{}]".format(y_label, i) for i in range(len(y_data[0]))]
                else:
                    labeling_format = ["result"]*len(y_data[0])

            for i in range(len(y_data[0])):
                plt_sub.plot(x_data, y_data[:, i], label="{}".format(labeling_format[i]))

        if x_interval:
            plt_sub.xticks(np.arange(x_lim[0], x_lim[-1], x_interval))
        if y_interval:
            plt_sub.yticks(np.arange(y_lim[0], y_lim[-1], y_interval))

        if x_label:
            plt_sub.xlabel(x_label)
        if y_label:
            plt_sub.ylabel(y_label)

        if average_range:
            if isinstance(average_range, list):
                assert len(average_range) == 2, \
                    "average_range must have 2 int"
                assert isinstance(average_range[0], int) and isinstance(average_range[1], int) and np.all(np.array(average_range) >= 0),\
                    "average_range elements must be positive int"

                avg_data = _moving_average(data=y_data,
                                           min_length=average_range[0], max_length=average_range[1])

            elif isinstance(float(average_range), float):
                avg_data = _moving_average(data=y_data,
                                           min_length=int(average_range),
                                           max_length=int(average_range))
            else:
                raise Exception("average_range must be int or list (len == 2)")

            plt_sub.plot(x_data, avg_data, "r", label="average")
            plt_sub.legend()

        if legend:
            if isinstance(legend, bool):
                plt_sub.legend()
            else:
                plt_sub.legend(**legend)

        plt_sub.grid(grid)

    def to_csv(self, filename, overwrite=False, epoch_data=True):
        """
        Stores csv file based on filename. Epoch data are stored as filename +"_e.csv"

        Args:

            filename (string): Filename of the string.
            overwrite (boolean): Overwrites if exist. Appends number if exist. Default is False.
            epoch_data (boolean): Stores epoch data if True. Default is True.

        Examples:

            >>> logger.to_csv("./test.csv", overwrite=True)

        """
        import csv
        import os

        assert isinstance(filename, str), "filename must be a string"
        assert filename.split(".")[-1] == "csv", "file name must be a csv"

        if not overwrite and os.path.exists(filename):
            exist = True
            file_i = 0

            while exist:

                file = ".".join(filename.split(".")[:-1])

                if "-" in file:
                    file = "-".join(file.split("-")[:-1])

                filename = "{}-{}.csv".format(file, file_i)
                exist = os.path.exists(filename)
                file_i += 1

        header, row_data = _to_csv_data(self._log_dic)

        with open(filename, mode="w") as f:
            row_data.insert(0, header)
            writer = csv.DictWriter(f, header)
            writer.writerows(row_data)

        if self._log_dic_epoch:
            filename_e = ".".join(filename.split(".")[:-1]) + "_e.csv"
            header_e, row_data_e = _to_csv_data(self._log_dic_epoch)

            with open(filename_e, mode="w") as f:
                row_data_e.insert(0, header_e)
                writer_e = csv.DictWriter(f, header_e)
                writer_e.writerows(header_e)

    def from_csv(self, filename):
        """
        Loads csv file based on filename. If file ends with '_e',csv file will be loaded as epoch data.

        Args:

            filename (string): Filename of the string.

        Examples:

            >>> logger.from_csv("./test.csv")

        """
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)

            m = np.minimum(len(filename) - 4, 6)

            if filename[-m:-4] != "_e":
                self._log_dic = {}
                for h in header:
                    self._log_dic[h] = []

                for row in reader:
                    for i, _ in enumerate(row):
                        self._log_dic[header[i]].append(eval(row[i]))
            else:
                self._log_dic_epoch = {}
                for h in header:
                    self._log_dic_epoch[h] = []

                for row in reader:
                    for i, _ in enumerate(row):
                        self._log_dic_epoch[header[i]].append(eval(row[i]))

            # for key in self._log_dic:
            #     self._log_dic[key]=np.array(self._log_dic[key])

    def set_description(self, msg):
        self._tqdm.set_description(msg)

    def update(self, num):
        if self._show_bar:
            self._tqdm.update(num)

    def close(self):
        self._tqdm.close()

    def _assert_logger_super(self):
        """
        this is used when creating new object
        """
        raise NotImplementedError(
            "Need to call super('class',self).__init__(log_key,record=True)  ('class':Class Name.)")


class SimpleLogger(Logger):
    """
    **Simple Logger Module**

    This class logs various data for each module.\n
     ``log_key`` , ``log_key_epoch`` argument must be a list of strings which exist in the algorithm.\n
     ``msg`` is required.　``msg_epoch`` is optional.

    Args:
        log_key(list): Logging values.
        log_key_epoch(list): Logging values at end of epoch.
        msg(string): Printing message for each iteration. Use curly braces '{}'.
        msg_epoch(string): Printing message for each epoch. Use curly braces '{}'.
        record(boolean): Keeps data for graph and csv. Default is True.
        record_episode_base(boolean): Keeps data when ``record`` is True and episode changes. Default is True.
        show_bar(boolean): Shows bar. Default is True.
        disable(boolean): Disables tqdm. Default is False.

    Examples:

        >>> logger = SimpleLogger(log_key = ["state","reward"] , msg="this is {state} reward:{reward}")
        >>> logger = SimpleLogger(log_key = ["state","reward"] , msg="this is {} reward:{}")

    """

    def __init__(self, log_key, log_key_epoch=None,
                 msg="", msg_epoch="", record=True,
                 record_episode_base=True,
                 show_bar=True, disable=False):

        log_key_epoch = [] if log_key_epoch is None else log_key_epoch
        super(SimpleLogger, self).__init__(log_key=log_key, log_key_epoch=log_key_epoch,
                                           record=record, record_episode_base=record_episode_base,
                                           show_bar=show_bar, disable=disable)
        self.message = msg
        self.message_epoch = msg_epoch
        assert len(log_key) == len(re.findall(r'\{+.?\}', msg)),\
            "log_key has {0} elements while message has {1} curly braces({2})".format(
            len(log_key), len(re.findall(r'\{+.?\}', msg)), "{ }")

        assert len(log_key_epoch) == len(re.findall(r'\{+.?\}', msg_epoch)),\
            "log_key has {0} elements while message has {1} curly braces({2})".format(
            len(log_key_epoch), len(re.findall(r'\{+.?\}', msg_epoch)), "{ }")

    def logger(self, **kwargs):
        """
        logs data
        """
        args = []
        for key in self._log_dic:
            args.append(kwargs[key])

        return self.message.format(*args)

    def logger_epoch(self, **kwargs):
        """
        logs epoch data
        """
        args = []
        for key in self._log_dic_epoch:
            args.append(kwargs[key])

        return self.message_epoch.format(*args)

    def logger_episode(self, **kwargs):
        """
        logs episode data
        """
