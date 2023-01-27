import sys
import os
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.getcwd()))
from utils import *


class AbstractPostprocessor(object):
    def __init__(self, minimal=None, maximal=None):
        super(AbstractPostprocessor, self).__init__()
        self.minimal = minimal
        self.maximal = maximal

    def update(self):
        pass

    def run(self, x):
        if self.minimal is not None and self.maximal is not None:
            return F.relu(self.minimal - x).pow(2) + F.relu(x - self.maximal).pow(2)
        if self.minimal is not None:
            return F.relu(self.minimal - x).pow(2)
        if self.maximal is not None:
            return F.relu(x - self.maximal).pow(2)
        return x


class ConstantPostprocessor(AbstractPostprocessor):
    def __init__(self, minimal=None, maximal=None):
        super(ConstantPostprocessor, self).__init__(minimal, maximal)

    def update(self):
        return


class ParabolPostprocessor(AbstractPostprocessor):
    def __init__(self, n_steps, m0=None, m1=None, M0=None, M1=None):
        super(ParabolPostprocessor, self).__init__()
        self.minimal = m0
        self.maximal = M0

        self.m0, self.m1 = m0, m1
        self.M0, self.M1 = M0, M1
        self.step_size = 1 / n_steps
        self.current = 0

    def update(self):
        self.current += self.step_size
        if self.minimal is not None:
            self.minimal = 4 * (self.m0 - self.m1) * self.current * (self.current - 1) + self.m0
        if self.maximal is not None:
            self.maximal = 4 * (self.M0 - self.M1) * self.current * (self.current - 1) + self.m0
        return


class LinearPostprocessor(AbstractPostprocessor):
    def __init__(self, n_steps, m0=None, m1=None, M0=None, M1=None):
        super(LinearPostprocessor, self).__init__()
        self.minimal = m0
        self.maximal = M0

        self.m0, self.m1 = m0, m1
        self.M0, self.M1 = M0, M1
        self.n_steps = n_steps
        self.step_size = 1 / n_steps
        self.current = 0
        self.done = False

    def run(self, x):
        self.update()
        if self.minimal is not None and self.maximal is not None:
            return F.relu(self.minimal - x).pow(2) + F.relu(x - self.maximal).pow(2)
        if self.minimal is not None:
            return F.relu(self.minimal - x).pow(2)
        if self.maximal is not None:
            return F.relu(x - self.maximal).pow(2)

    def update(self):
        if self.done:
            return
        self.current += 1
        if self.minimal is not None:
            self.minimal += (self.m1 - self.m0) * self.step_size
        if self.maximal is not None:
            self.maximal += (self.M1 - self.M0) * self.step_size
        if self.current >= self.n_steps:
            self.done = True
        return
