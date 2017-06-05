
'''This module implements the settings used in the experiments of Section 6 from [1].

[1] Durand, A., Gagne, C.: Estimating Quality in Multi-Objective Bandits Optimization.
    https://arxiv.org/abs/1701.01095
'''

import numpy


class MultiBernoulli:
    def __init__(self, means, randomseed=None):
        self.random = numpy.random.RandomState(randomseed)
        self.means = means
        self.nb_objectives = means.shape[1]
    
    def play(self, action):
        return self.random.rand(self.nb_objectives) < self.means[action]


class MultivariateNormal:
    def __init__(self, means, cov=[[0.1, 0.05], [0.05, 0.1]], randomseed=None):
        self.random = numpy.random.RandomState(randomseed)
        self.means = means
        self.cov = cov
    
    def play(self, action):
        return self.random.multivariate_normal(self.means[action], self.cov)
