import numpy as np

from ..operator import operator as opt
from .base import BasePop

EPS = 1e-8

class DEPop(object):
    """ DE population """
    def __init__(self, pop, fit, lb=-float('inf'), ub=float('inf')):

        # init pop
        self.pop = pop
        self.fit = fit
        self.gen_pop = None
        self.gen_fit = None
        self.new_pop = None
        self.new_fit = None

        # states
        self.num_iter = 0
        
        # parameters
        self.pop_size = None        
        self.gen_size = None

    def remap(self, samples):
        return opt.mirror_map(samples, self.lb, self.ub)

    def eval(self, e):
        self.gen_pop = e()

    def select(self):
        """ Select 'new_pop' and 'new_fit' """
        raise NotImplementedError

    def generate(self):
        """ Generate offsprings """
        raise NotImplementedError

    def adapt(self):
        """ Adapt new states """
        raise NotImplementedError

    def update(self):
        """ Update pop and states """
        raise NotImplementedError

    def evolve(self):