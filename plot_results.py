
import argparse
import os

from matplotlib import pyplot

import numpy


pyplot.rc("text", usetex=True)
pyplot.rc("font", family="serif", serif="cm10", size=10)
pyplot.rc("legend", framealpha=0, fontsize=10)

parser = argparse.ArgumentParser()
parser.add_argument("setting", choices=["bernoulli", "normal"])
parser.add_argument("preference", choices=["linear", "econstraint"])
parser.add_argument("nb_repeat", type=int)
parser.add_argument("--path", type=str, default="./results")
parser.add_argument("--output", type=str, default=None)
args = parser.parse_args()


def extract(inpath, nb_repeat):
    results = []
    for rep in range(nb_repeat):
        result = numpy.loadtxt(os.path.join(inpath, str(rep)), delimiter=",")
        results.append(result)
    return results


def plot(results, color, label):
    mean = numpy.mean(results, 0)
    episodes = numpy.arange(0, 10000, 100)
    pyplot.plot(episodes, mean[episodes], "-", color=color, linewidth=2, label=label)
    for res in results:
        pyplot.plot(episodes, res[episodes], ":", color=color, linewidth=1, alpha=0.4)


def get_n_best(results):
    last = results[:, :, -1]
    best = numpy.argmin(last, axis=0)
    return [numpy.sum(best == i) for i in range(last.shape[0])]


path = os.path.join(args.path, args.setting, args.preference)

confs = [("MVN_TS", "tab:blue", "MVN-TS"),
         ("Gaussian_TS", "tab:orange", "Gaussian-TS")]

pyplot.figure(figsize=(3, 2))
results = []
for algo, color, label in confs:
    results.append(extract(os.path.join(path, algo), args.nb_repeat))
    plot(results[-1], color, label)
# results is a NB_ALGO x NB_REPEAT x NB_EPISODES matrix
results = numpy.array(results)

ylims = {("bernoulli", "linear"): (0, 200),
         ("bernoulli", "econstraint"): (0, 400),
         ("normal", "linear"): (0, 400),
         ("normal", "econstraint"): (0, 800)}

handles, labels = pyplot.gca().get_legend_handles_labels()
pyplot.legend(handles[::-1], labels[::-1], loc="upper left")

pyplot.ylabel("Cumulative regret")
pyplot.xlabel("Episodes")
pyplot.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

if args.output is None:
    pyplot.show()
else:
    figname = os.path.join(args.output, args.setting + "_" + args.preference+".pdf")
    pyplot.savefig(figname, bbox_inches="tight", pad_inches=0.03, frameon=None)

