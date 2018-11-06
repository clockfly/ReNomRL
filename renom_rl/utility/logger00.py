from tqdm import tqdm


class LogObject(object):

    def __init__(self):
        pass

    def set_description(self,msg):
        pass

    def update(self,i):
        pass

    def refresh(self):
        pass

    def close(self):
        pass



class TQLogger(LogObject):
    def __init__(self,max_length,msg="",var=[]):
        self.tq=tqdm(range(max_length))
        self.msg=msg
        self.var_list=var

    def set_msg(self, msg="", var=[]):
        assert msg.count("}")==len(var), "the message has {} but the variable is set to {}".format(msg.count("}"),len(var))
        self.msg=msg
        self.var_list=var

    def list(self):
        string = "["+",".join(self.var_list)+"]"
        return string

    def log(self,list=[]):
        self.tq.update(1)
        self.tq.set_description(self.msg.format(*list))

    def format(self,msg,list=[]):
        assert msg.count("}")==len(var), "the message has {} but the variable is set to {}".format(msg.count("}"),len(var))
        self.tq.set_description(msg.format(*list))


    def set_description(self,msg):
        self.tq.set_description(self.msg)

    def update(self,i):
        self.tq.update(i)

    def refresh(self):
        self.tq.refresh()

    def close(self):
        self.tq.close()



class PrintLogger(LogObject):
    def __init__(self,msg="",var=[]):
        self.print_obj=print
        self.msg=msg
        self.var_list=var

    def set_msg(self, msg="", var=[]):
        assert msg.count("}")==len(var), "the message has {} but the variable is set to {}".format(msg.count("}"),len(var))
        self.msg=msg
        self.var_list=var

    def list(self):
        string = "["+",".join(self.var_list)+"]"
        return string

    def log(self,list=[]):
        self.print_obj(self.msg.format(*list))

    def format(self,msg,list=[]):
        assert msg.count("}")==len(var), "the message has {} but the variable is set to {}".format(msg.count("}"),len(var))
        self.tq.set_description(msg.format(*list))



class LogSelector(object):
    def __init__(self,mode=""):
        dic={
        "TQLogger":TQLogger,
        "PrintLogger":PrintLogger,
        }
