import numpy as np
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
from copy import copy, deepcopy


def _moving_average(data, min_length, max_length):
    """
    Average moving length.
    """

    res = []
    for i in range(len(data)):
        min_i = np.maximum(i - min_length,0)
        max_i = np.minimum(i + max_length + 1,len(data))

        avg = np.sum(data[min_i:max_i],axis=0)/(max_i-min_i)
        res.append(avg)

    return np.array(res)


def _pass_logger():
    pass


def _remove_col(func):
    """
    This function is used to remove `:` for tqdm.
    tqdm ver.4.19 work.
    """
    def inner(*args,**kwargs):
        return func(*args,**kwargs)[:-2]

    return inner

def _log_decorator_iter(self,log_func):
    """
    This function will decorate logging function.
    Do not delete this function.
    """
    def _decorator(**kwargs):

        self.update(1)

        kwargs=deepcopy(kwargs)

        log_msg = log_func(**kwargs)

        if log_msg:
            self.set_description(log_msg)

        if self.record:
            for key in self.log_dic:
                if key in kwargs:
                    self.log_dic[key].append(kwargs[key])



    return _decorator

def _log_decorator_epoch(self,log_func):
    """
    This function will decorate logging function for epoch run.
    Do not delete this function. Note there is no update.
    """
    def _decorator2(**kwargs):

        kwargs=deepcopy(kwargs)

        log_msg = log_func(**kwargs)

        if log_msg:
            self.set_description(log_msg)

        if self.record:
            for key in self.log_dic:
                if key in kwargs:
                    self.log_dic[key].append(kwargs[key])

    return _decorator2


class LoggerMeta(type):
    def __call__(cls, *args, **kwargs):
        self = cls.__new__(cls, *args, **kwargs)
        cls.__init__(self, *args, **kwargs)
        self._assert_logger_super()
        return self




class Logger(object, metaclass=LoggerMeta):
    """
    **Logger Module**

    This class logs various data of each module.
    By setting `log_key`, this class will record data based on log_key for every iteration.
    `log_key` argument must be a list of strings which exist in the algorithm.
    `logger(**log)` function returns at every iter. (Overwriting required.)
    Note that you must also call super class constructor(super().__init__) when initializing.

    Args:

        log_key(list): Logging values.
        record(boolean): Keeps data for graph and csv. Default is True.
        show_bar(boolean): Shows bar. Default is True.
        disable(boolean): Disables tqdm. Default is False.

    Examples:
        >>> from renom_rl.utility.logger import Logger
        >>> import numpy as np
        >>>
        >>> class Original(Logger):
        ...     def __init__(self,log_key,record=True):
        ...         super(Original,self).__init__(log_key,record=True)
        ...
        ...         self.reward=0
        ...         self.state=0
        ...         self.terminal=0
        ...
        ...     def logger(self,**log):
        ...         self.state = log["state"]
        ...         self.reward = log["reward"]
        ...         self.terminal= log["terminal"]
        ...
        ...         if self.terminal:
        ...             print("terminal")
        ...
        ...         return "{}/{}".format(self.reward , self.state)
        ...
        >>> original_logger=Original(["reward"])
        >>> s=0
        >>> length=30
        >>> original_logger.start(length)
        >>> for i in range(length):
        ...     s += 1
        ...     r = s + np.random.randint(-5,6)
        ...     t = np.random.randint(0,1)
        ...
        ...     original_logger.logger(state=s,reward=r,terminal=t)
        ...
        >>> original_logger.close()
        30/30: 100%|██████████████████████████████████| 30/30 [00:00<00:00, 4865.41it/s]

    """

    def __init__(self,log_key=None, record=True, show_bar=True, disable=False):

        log_key = [] if log_key is None else log_key
        assert isinstance(log_key,list), "log_var must be a list."
        assert "env" not in log_key or "network" not in log_key, "do not record env or network object."

        log_dic={}
        for key in log_key:
            log_dic[key]=[]

        self.log_dic = log_dic
        self.record = record
        self._assert_logger_super = _pass_logger
        self.logger = _log_decorator_iter(self,self.logger)
        self.logger_epoch = _log_decorator_epoch(self,self.logger_epoch)
        self.show_bar = show_bar
        self.tqdm = None
        self.disable = disable


    def start(self,length):
        """
        Initializes tqdm.

        Args:

            length(float): length of tqdm.

        """
        assert length, "set argument `length`"
        if not self.show_bar:
            self.tqdm = tqdm(range(1),total=1,bar_format="-{desc}",disable=self.disable)
            self.tqdm.__repr__ = remove_col(self.tqdm.__repr__)
        else:
            self.tqdm = tqdm(range(length),disable=self.disable)

    def _key_check(self,existing_key_input):
        """
        Checks key. This will be called from each algorithm.
        """
        for key in self.log_dic:
            assert key in existing_key_input, "{} does not exist as logging key in this module. Reset log_key.".format(key)

    def reset(self):
        """
        Resets logged data.
        """
        for key in self.log_dic:
            self.log_dic[key]=[]

    def logger(self, **kwargs):
        """
        This function will be called for every iteration.
        Override this function when creating custom logger.
        """

        raise NotImplementedError("Please override ``logger`` method.")

    def logger_epoch(self, **kwargs):
        """
        This function will be called when 1 epoch is done.
        Override this function when creating custom logger.
        """
        pass


    def result(self,*args):
        """
        Returns dictionary of result. If argument is blank, then all output will be shown.

        Args:
            *args(string): Strings of arguments.

        Returns:
            (dict): Dictionary of rewards.

        Examples:
            >>> original_logger.result("reward")
            {'reward': [2, 3, 1, 2, 6, 3, 10, 7, 8, 11, 15, 8, 17, 15, 14, 19, 21, 16, 22, 24, 24, 25, 26, 24, 24, 25, 32, 31, 31, 30]}

        """
        if args:
            res={}
            for key in args:
                assert key in self.log_dic,"no such key as {} in log_key".format(key)
                res[key]=self.log_dic[key]
            return res
        else:
            return self.log_dic


    def graph(self,y_key,x_key=None,x_lim=None,y_lim=None,
              x_interval=0,y_interval=0,figsize=None,
              dpi=100,average_range=0,grid=True):

        """
        Shows plot from recorded data. Keys must be from log_key. if `x_key` is None,
        y_key will be plot based on its length. Note that this function is focused on
        quick view so if detailed view is required, save data as csv.(refer `to_csv` function)

        Args:
            y_key (string): Key for Y (vertical) axis. 2-D data is allowed.
            x_key (string): Key for X (horizontal) axis. This must be 1-D data. Default is None.
            x_lim (list): [min,max] range for X axis. Default is min,max of x_key data.
            y_lim (list): [min,max] range for Y axis. Default is min,max of y_key data.
            x_interval (float): Interval of ticks for X axis. Default is None (ticks: auto).
            y_interval(float): Interval of ticks for X axis. Default is None (ticks: auto).
            figsize (tuple): When (a,b) is input, plot increase to a*x_axis, b*y_axis.
            dpi (int): Digital Pixel Image. Default is 100.
            average_length (int or list): Creates average plot in [i - min, i + max] range \
                                    (i being plotted point), when [min,max] is set. If int \
                                    type is set, it becomes [average_length,average_length].\
                                     Default is 0.
            grid (boolean): Shows grid based on ticks. Default is True.

        Examples:
            >>> original_logger.graph(y_key="reward",figsize=(6,6),dpi=100,average_range=[1,1])

        """

        # x_data, y_data
        assert y_key in self.log_dic,\
           "set axis to keys set in log_key variable"

        y_data=np.array(self.log_dic[y_key])

        if x_key is not None:
            assert x_key in self.log_dic,\
                 "set axis to keys set in log_key variable"
            x_data=np.array(self.log_dic[x_key])
        else:
            x_data=None

        # creating custom graph
        self.graph_custom(y_data=y_data,x_data=x_data,y_label=y_key,x_label=x_key, x_lim=x_lim,y_lim=y_lim,
                  x_interval=x_interval,y_interval=y_interval,figsize=figsize,
                  dpi=dpi,average_range=average_range,grid=grid)

    #for custom graph
    def graph_custom(self,y_data,x_data=None,y_label="",x_label="", x_lim=None,y_lim=None,
              x_interval=0,y_interval=0,figsize=None,
              dpi=100,average_range=0,grid=True):
        """
        This function allows users to quickly create graph when custom creating own data.
        refer `graph` for other arguments.

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

        y_data = np.array(y_data)
        x_data = x_data if x_data is not None else np.arange(len(y_data))

        assert len(np.shape(x_data)) <= 1 and len(np.shape(y_data)) <= 2,\
            "key dimension conditions are x_data <= 1 and y_data <= 2 when plotting"

        if figsize:
            plt.figure(figsize=figsize,dpi=dpi)

        if x_lim:
            assert isinstance(x_lim,list) and len(x_lim)==2,\
                  "x_lim must be a [min,max] structure"

        if y_lim:
            assert isinstance(y_lim,list) and len(y_lim)==2,\
                  "y_lim must be a [min,max] structure"


        x_lim = x_lim if x_lim else [np.min(x_data),np.max(x_data)]
        y_lim = y_lim if y_lim else [np.min(y_data),np.max(y_data)]

        plt.xlim(x_lim)
        plt.ylim(y_lim)

        if len(np.shape(y_data))==1:
            plt.plot(x_data,y_data,"b",label="result")
        if len(np.shape(y_data))==2:
            for i in range(len(y_data[0])):
                plt.plot(x_data,y_data[:,i],label="{}[{}]".format(y_label,i))


        if x_interval: plt.xticks(np.arange(x_lim[0], x_lim[-1], x_interval))
        if y_interval: plt.yticks(np.arange(y_lim[0], y_lim[-1], y_interval))

        if x_label: plt.xlabel(x_label)
        if y_label: plt.ylabel(y_label)

        if average_range:
            if isinstance(average_range,list):
                assert len(average_range)==2, \
                        "average_range must have 2 int"
                assert isinstance(average_range[0],int) and isinstance(average_range[1],int) and np.all(np.array(average_range)>=0),\
                        "average_range elements must be positive int"

                avg_data = _moving_average(data = y_data,
                            min_length = average_range[0], max_length=average_range[1])

            elif isinstance(float(average_range),float):
                avg_data = _moving_average(data = y_data,
                            min_length = int(average_range),
                            max_length = int(average_range))
            else:
                assert Exception("average_range must be int or list (len == 2)")

            plt.plot(x_data,avg_data,"r",label="average")
            plt.legend()

        plt.grid(grid)
        plt.show()


    def to_csv(self,filename,overwrite=False):
        """
        Stores csv file based on filename.

        Args:

            filename (string): Filename of the string.
            overwrite (boolean): Overwrites if exist. Appends number if exist. Default is False.

        Examples:

            >>> logger.to_csv("./test.csv", overwrite=True)

        """
        import csv
        import os

        assert isinstance(filename,str),"filename must be a string"
        assert filename.split(".")[-1]=="csv", "file name must be a csv"

        if not overwrite and os.path.exists(filename):
            exist=True
            file_i=0

            while exist:

                file = ".".join(filename.split(".")[:-1])

                if "-" in file:
                    file = "-".join(file.split("-")[:-1])

                filename = "{}-{}.csv".format(file,file_i)
                exist = os.path.exists(filename)
                file_i += 1


        key_list = self.log_dic.keys()
        header = {val:val for val in key_list}
        first_key = next(iter(self.log_dic))
        data_length = len(self.log_dic[first_key])

        row_data=[]

        for i in range(data_length):
            row_data.append({key_i: \
                self.log_dic[key_i][i].tolist() if isinstance(self.log_dic[key_i][i],np.ndarray) \
                else self.log_dic[key_i][i] \
                    for key_i in self.log_dic})
            # row_data.append({key_i:self.log_dic[key_i][i] for key_i in self.log_dic})


        with open(filename, mode="w") as f:
            row_data.insert(0, header)
            writer = csv.DictWriter(f, header)
            writer.writerows(row_data)


    def from_csv(self,filename):
        """
        Loads csv file based on filename.

        Args:

            filename (string): Filename of the string.

        Examples:

            >>> logger.from_csv("./test.csv")

        """
        with open(filename, 'r') as f:
            reader = csv.reader(f,delimiter=',')
            header = next(reader)

            self.log_dic={}
            for h in header:
                self.log_dic[h]=[]

            for row in reader:
                for i,_ in enumerate(row):
                    self.log_dic[header[i]].append(eval(row[i]))

            # for key in self.log_dic:
            #     self.log_dic[key]=np.array(self.log_dic[key])


    def  set_description(self,msg):
        self.tqdm.set_description(msg)

    def update(self,num):
        if self.show_bar:
            self.tqdm.update(num)

    def close(self):
        self.tqdm.close()

    def _assert_logger_super(self):
        """
        this is used when creating new object
        """
        raise NotImplementedError("Need to call super('class',self).__init__(log_key,record=True)  ('class':Class Name.)")







class SimpleLogger(Logger):
    """
    **Simple Logger Module**

    This class logs various data of each module.
    `log_key` argument must be a list of strings which exist in the algorithm.
    `msg` is required.

    Args:

        log_key(list): logging values.
        record(boolean): keeps data for graph and csv. Default is True.

    Examples:

        >>> logger = SimpleLogger(log_key = ["state","reward"] , msg="this is {} reward:{}")

    """
    def __init__(self, log_key, msg="",record = True,show_bar=True, disable=False):
        super(SimpleLogger,self).__init__(log_key, record, show_bar, disable)
        self.message = msg
        assert len(log_key)==len(re.findall(r'\{+.?\}',msg)),"log_key has {0} elements while message has {1} curly braces({2})".format(len(log_key),len(re.findall(r'\{+.?\}',msg)),"{ }")

    def logger(self, **kwargs):
        args=[]
        for key in self.log_dic:
            args.append(kwargs[key])

        return self.message.format(*args)
