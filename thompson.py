
'''This module implements algorithms from the following papers.

[1] Durand, A., Gagne, C.: Estimating Quality in Multi-Objective Bandits Optimization.
    https://arxiv.org/abs/1701.01095
[2] Agrawal, S., Goyal, N.: Further optimal regret bounds for Thompson Sampling.
    In: Proceedings of the 16th International Conference on Artificial Intelligence and
    Statistics (AISTATS). pp. 99–107 (2013)
'''

import numpy


class MVN_TS:
    '''This class implements the Thompson sampling from MVN priors algorithm (Alg. 2 from [1]).
    '''
    def __init__(self, nb_actions, nb_objectives):
        self.data = [[] for i in range(nb_actions)]
        self.n = numpy.zeros(nb_actions)
        self.i_d = numpy.identity(nb_objectives)
        self.mu1 = [numpy.zeros(nb_objectives) for i in range(nb_actions)]
        self.sigma1 = [self.i_d for i in range(nb_actions)]
    
    def update(self, action, outcome):
        self.data[action].append(outcome)
        self.n[action] += 1
        data, n = self.data[action], self.n[action]
        mu_hat = numpy.mean(data, 0)
        sigma1 = numpy.linalg.inv(self.i_d + n * self.i_d)
        mu1 = numpy.dot(sigma1, numpy.dot(n * self.i_d, mu_hat))
        self.mu1[action] = mu1
        self.sigma1[action] = sigma1
    
    def get_options(self):
        options = [numpy.random.multivariate_normal(mu1, sigma1)
                    for mu1, sigma1 in zip(self.mu1, self.sigma1)]
        return options
    
    def get_means(self):
        return [numpy.mean(data, 0) if data else 0 for data in self.data]


class Gaussian_TS:
    '''This class implements the Thompson sampling from Gaussian priors algorithm (Alg. 3 from [1])
    taken from [2].
    '''
    def __init__(self, nb_actions):
        self.data = [[] for i in range(nb_actions)]
        self.n = numpy.zeros(nb_actions)
        self.mu1 = [0 for i in range(nb_actions)]
        self.sigma1 = [1 for i in range(nb_actions)]
    
    def update(self, action, outcome):
        self.data[action].append(outcome)
        self.n[action] += 1
        data, n = self.data[action], self.n[action]
        sigma1 = 1 / (n + 1)
        mu_hat = numpy.mean(data, 0)
        mu1 = numpy.sum(data, 0) / (n + 1)
        self.mu1[action] = mu1
        self.sigma1[action] = sigma1
    
    def get_options(self):
        return numpy.random.normal(self.mu1, self.sigma1)
    
    def get_means(self):
        return [numpy.mean(data, 0) if data else 0 for data in self.data]

