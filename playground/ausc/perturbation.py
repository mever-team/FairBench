import random
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import fairbench as fb


def accuracy(yhat_, y):
    return int((yhat_ == y).mean() * 100) / 100


def random_flip(yhat_, candidates, intensity):
    part = np.copy(yhat_)
    for i in random.sample(candidates, intensity):
        part[i] = 1 - part[i]
    return part


def perturbations(
    yhat_, y, sensitive, fairness, intensity=1, n=10, eq=lambda a, b: a == b
):
    candidates = [i for i in range(len(yhat_)) if eq(yhat_[i], y[i]) and sensitive[i]]
    fair = list()
    accs = list()
    for i in range(n):
        try:
            intense = random.randint(1, intensity)
            part = random_flip(yhat_, candidates, intensity=intense)
        except ValueError:
            continue
        fair.append(fairness(part, y, sensitive=sensitive))
        accs.append(accuracy(part, y))
    return fair, accs


def scatter(accs, fair, **kwargs):
    plt.scatter(accs, fair, **kwargs)


def line(accs, fairs, label, start_acc=None, start_fair=None, show=True, **kwargs):
    assert start_acc is not None
    assert start_fair is not None
    map = dict()
    for a, f in zip(accs, fairs):
        map[a] = min(map.get(a, float("inf")), f)
    accs = sorted(map.keys())
    fairs = [map[a] for a in accs]
    accs = np.array(accs)
    fairs = np.array(fairs)  # - np.abs(np.array(accs) - start_acc)
    fairs = start_fair - np.abs(start_fair - fairs)

    ausc = np.trapezoid(fairs, accs) / (max(accs) - min(accs))
    ausc = float(ausc)
    if show:
        plt.plot(accs, fairs, **kwargs, label=label + f" (area {ausc:.3f})")
    return ausc


def run(yhat, y, sensitive, fairness, n=10, plot=line, step=1, **kwargs):
    balance_range = len(y)
    fair, accs = list(), list()
    sens = [sensitive, 1 - sensitive]
    eqs = [
        # lambda a, b: a == 1,
        # lambda a, b: b == 1,
        # lambda a, b: a == 0,
        # lambda a, b: b == 0,
        lambda a, b: a == b,
        lambda a, b: a == b and a == 1,
        lambda a, b: a == b and a == 0,
        lambda a, b: a == b and b == 1,
        lambda a, b: a == b and b == 0,
        lambda a, b: a != b,
        lambda a, b: a != b and a == 1,
        lambda a, b: a != b and a == 0,
        lambda a, b: a != b and b == 1,
        lambda a, b: a != b and b == 0,
    ]
    for balance in tqdm(range(-balance_range, balance_range + 1, step)):
        for sensitive in sens:
            for eq in eqs:
                res = perturbations(
                    yhat,
                    y,
                    fairness=fairness,
                    n=n,
                    intensity=balance_range + 1,
                    sensitive=sensitive,
                    eq=eq,
                )
                fair.extend(res[0])
                accs.extend(res[1])
    return plot(
        accs,
        fair,
        start_acc=accuracy(yhat, y),
        start_fair=fairness(yhat, y, sensitive=sensitive),
        **kwargs,
    )


def prule(yhat_, y, sensitive):
    cl1 = fb.measures.pr(yhat_, sensitive=sensitive) | float
    cl2 = fb.measures.pr(yhat_, sensitive=1 - sensitive) | float
    return min(cl1, cl2) / max(cl1, cl2)


def dfpr(yhat_, y, sensitive):
    cl1 = fb.measures.tpr(yhat_, y, sensitive=sensitive) | float
    cl2 = fb.measures.tpr(yhat_, y, sensitive=1 - sensitive) | float
    return 1 - abs(cl1 - cl2)


def dfnr(yhat_, y, sensitive):
    cl1 = fb.measures.tnr(yhat_, y, sensitive=sensitive) | float
    cl2 = fb.measures.tnr(yhat_, y, sensitive=1 - sensitive) | float
    return 1 - abs(cl1 - cl2)


def eq_odds(yhat_, y, sensitive):
    return (dfpr(yhat_, y, sensitive) + dfnr(yhat_, y, sensitive)) / 2


def pred_parity(yhat_, y, sensitive):
    cl1 = fb.measures.ppv(yhat_, y, sensitive=sensitive) | float
    cl2 = fb.measures.ppv(yhat_, y, sensitive=1 - sensitive) | float
    return 1 - abs(cl1 - cl2)
