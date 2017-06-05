
'''This module contains the linear and e-constraint preference functions used in the
experiments of Section 6 from [1].

[1] Durand, A., Gagne, C.: Estimating Quality in Multi-Objective Bandits Optimization.
    https://arxiv.org/abs/1701.01095
'''


def linear(x):
    weights = [0.4, 0.6]
    return sum([a * x_i for a, x_i in zip(weights, x)])


def econstraint(x):
    target = 1
    thresholds = [0.5, 0]
    for i, x_i in enumerate(x):
        if i != target:
            if x_i < thresholds[i]:
                return 0
    return x[target]
