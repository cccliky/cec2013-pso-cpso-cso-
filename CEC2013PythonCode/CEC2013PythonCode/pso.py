import numpy as np

from matplotlib import pyplot as plt
import time
import numpy as np
import math

class PSO(object):
    def __init__(self, pop, max_iter, lb, ub, dim, objective_function, w=0.8, c1=0.5, c2=0.5):
        self.objective_function = objective_function
        self.w = w
        self.cp, self.cg = c1, c2
        self.pop = pop
        self.dim = dim
        self.max_iter = max_iter

        self.has_constraints = not (lb is None and ub is None)
        self.lb = -np.ones(self.dim) if lb is None else np.array(lb)
        self.ub = np.ones(self.dim) if ub is None else np.array(ub)
        assert self.dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.dim))
        v_high = self.ub - self.lb
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.dim))
        self.Y = np.zeros([self.pop], )
        self.Y = np.array([self.objective_function(pos) for pos in self.X])

        self.pbest_x = self.X.copy()
        self.pbest_y = self.Y.copy()
        self.gbest_x = np.zeros((1, self.dim))
        self.gbest_y = np.inf
        self.gbest_y_hist = []
        self.update_gbest()

        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}

    def update_V(self):
        r1 = np.random.rand(self.pop, self.dim)
        r2 = np.random.rand(self.pop, self.dim)
        self.V = self.w * self.V + \
                 self.cp * r1 * (self.pbest_x - self.X) + \
                 self.cg * r2 * (self.gbest_x - self.X)

    def update_X(self):
        self.X = self.X + self.V

        if self.has_constraints:
            self.X = np.clip(self.X, self.lb, self.ub)

    def update_pbest(self):
        self.pbest_y = self.pbest_y.reshape((40, 1))
        self.Y = self.Y.reshape((40, 1))
        self.pbest_x = np.where(self.pbest_y > self.Y, self.X, self.pbest_x)
        self.pbest_y = np.where(self.pbest_y > self.Y, self.Y, self.pbest_y)

    def update_gbest(self):
        if self.gbest_y > self.Y.min():
            self.gbest_x = self.X[self.Y.argmin(), :].copy()
            self.gbest_y = self.Y.min()

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['Y'].append(self.Y)
        self.record_value['V'].append(self.V)

    def optimize(self):
        self.gbest_y_hist = []
        for iternum in range(self.max_iter):
            self.update_V()
            self.recorder()
            self.update_X()
            self.Y = np.array([self.objective_function(pos) for pos in self.X])
            self.update_pbest()
            self.update_gbest()
            print("the number of %d is %.1f" % (iternum, self.gbest_y))

            self.gbest_y_hist.append(self.gbest_y)
        #print("the number of %d is %.1f" % (iternum, self.gbest_y))
        return self.gbest_y_hist


def PS(pop, max_iter, lb, ub, dim, objective_function):
    optimizer = PSO(pop,max_iter,lb,ub,dim,objective_function, w=0.8, c1=0.5, c2=0.5)
    curve = optimizer.optimize()
    return curve