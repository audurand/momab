
import numpy


def run_somab(setting, algorithm, nb_episodes=10000):
    cumul_regret = [0]
    options = numpy.random.rand(len(means))
    for t in range(nb_episodes):
        # select a_t
        a_t = numpy.argmax(options)
    
        # play a_t and observe outcome
        z_t = setting.play(a_t)

        # cumulate regret
        cumul_regret.append(cumul_regret[-1] + regrets[a_t])

        # update knowledge
        algorithm.update(a_t, f(z_t))
    
        options = algorithm.get_options()
    return cumul_regret


def run_momab(setting, algorithm, nb_episodes=10000):
    cumul_regret = [0]
    o_t = numpy.random.randint(len(means))
    for t in range(nb_episodes):
        # select a_t
        a_t = o_t

        # play a_t and observe outcome
        z_t = setting.play(a_t)

        # cumulate regret
        cumul_regret.append(cumul_regret[-1] + regrets[a_t])

        # update knowledge
        algorithm.update(a_t, z_t)

        # get feedback from expert user
        options = algorithm.get_options()
        o_t = numpy.argmax([f(o) for o in options])
    return cumul_regret


if __name__ == "__main__":
    import argparse
    import os

    from functions import linear, econstraint
    from settings import MultiBernoulli, MultivariateNormal
    from thompson import Gaussian_TS, MVN_TS

    parser = argparse.ArgumentParser()
    parser.add_argument("algo", choices=["Gaussian_TS", "MVN_TS"])
    parser.add_argument("setting", choices=["bernoulli", "normal"])
    parser.add_argument("preference", choices=["linear", "econstraint"])
    parser.add_argument("start", type=int)
    parser.add_argument("repeat", type=int)
    parser.add_argument("--path", type=str, default="./results")
    args = parser.parse_args()

    if args.setting == "bernoulli":
        set_cls = MultiBernoulli
    elif args.setting == "normal":
        set_cls = MultivariateNormal
    else:
        raise Exception("Unknown setting "+args.setting)

    f = eval(args.preference)

    path = os.path.join(args.path, args.setting, args.preference, args.algo)
    os.makedirs(path, exist_ok=True)

    means = numpy.loadtxt("means_2d", delimiter=",")

    values = [f(m) for m in means]
    preferred_idx = numpy.argmax(values)
    regrets = [values[preferred_idx] - v for v in values]

    if args.algo == "Gaussian_TS":
        for rep in range(args.start, args.start+args.repeat):
            setting = set_cls(means, randomseed=rep)
            algorithm = Gaussian_TS(means.shape[0])
            cumul_regret = run_somab(setting, algorithm)
            numpy.savetxt(os.path.join(path, str(rep)), cumul_regret, delimiter=",")
    else:
        for rep in range(args.start, args.start+args.repeat):
            setting = set_cls(means, randomseed=rep)
            algorithm = eval(args.algo)(*means.shape)
            cumul_regret = run_momab(setting, algorithm)
            numpy.savetxt(os.path.join(path, str(rep)), cumul_regret, delimiter=",")

