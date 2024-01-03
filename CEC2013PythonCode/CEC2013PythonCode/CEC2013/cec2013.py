from CEC2013.functions import *

from CEC2013PythonCode.CEC2013.functions import CEC_functions


class cec2013:
    num_of_func = 28
    def __init__(self,dim,func_id):
        self.D = dim
        self.N = 50
        self.bench = CEC_functions(self.D)
        self.func_id = func_id
        self.max = 100
        self.min = -100
        self.MAXFES = self.D * 10000


    def get_Parameters(self):
        map = {'MAX':self.max, 'MIN':self.min, 'N':self.N, 'D':self.D, 'MAXFES':self.MAXFES}
        return map

    def func(self, x):
        return self.bench.Y(x, self.func_id)

    def end(self):
        return 0


