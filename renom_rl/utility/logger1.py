from tqdm import tqdm


class LogObject(object):

    def __init__(self):
        self.var=[]
        pass

    def log(self,*var):
        self.var=var
        self.run()
        pass

    def reset(self):
        pass

    def close(self):
        pass



class TQLogger(LogObject):

    def __init__(self):
        self.tq=tqdm

    def log(self,*var):
        self.tq.update(1)
        msg="greedy{:.4f} epoch {:04d} loss {:5.4f} rewards in epoch {:4.3f} episode {:04d} rewards in episode {:4.3f}."
        self.tq.set_description(msg.format(*var[]))

    def log(self,*args):
        self.tq.update(1)
        msg = "epoch {:04d} loss {:5.4f} rewards in epoch {:4.3f} episode {:04d} rewards in episode {:4.3f}."
        res=self.calculate(args)
        self.tq.set_description(self.msg.format(*res))

    def calculate(self,args):
        args




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
