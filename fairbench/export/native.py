from fairbench.forks.fork import Fork, Forklike
from fairbench.reports.accumulate import todict
from matplotlib import pyplot as plt
import json
from fairbench.forks.explanation import tofloat
from fairbench.forks import ExplanationCurve


def _is_fork_of_dicts(report):
    return isinstance(report[next(iter(report))], dict)


def tojson(report: Fork):
    if isinstance(report, dict):  # includes Forklike
        report = todict(**report)
    if isinstance(report, dict):  # if it's still a Forklike
        report = Fork(report)
    assert isinstance(report, Fork)
    report = {
        k: v.branches() if isinstance(v, Fork) else v
        for k, v in report.branches().items()
    }
    data = dict()
    if not _is_fork_of_dicts(report):
        report = {k: {"": v} for k, v in report.items()}
    data["header"] = ["Metric"] + [key for key in report]
    for value in report.values():
        for metric in value:
            if metric not in data:
                data[metric] = list()
            if isinstance(value[metric], ExplanationCurve):
                data[metric].append(
                    {
                        "x": [x for x in value[metric].x],
                        "y": [y for y in value[metric].y],
                    }
                )
            else:
                data[metric].append(tofloat(value[metric]))
    return json.dumps(data)


def describe(report: Fork, spacing: int = 15):
    report = json.loads(tojson(report))
    ret = ""
    if report["header"]:
        ret += " ".join([entry.ljust(spacing) for entry in report["header"]]) + "\n"
    for metric in report:
        if metric != "header":
            ret += (
                " ".join(
                    [metric.ljust(spacing)]
                    + [f"{entry:.3f}".ljust(spacing) for entry in report[metric]]
                )
                + "\n"
            )
    print(ret)


def visualize(report: Fork, hold: bool = False, xrotation: int = 0):
    report = json.loads(tojson(report))

    i = 1
    for metric in report:
        if metric != "header":
            plt.subplot(2, len(report) // 2, i)
            barplots = False
            for j, case in enumerate(report["header"][1:]):
                if isinstance(report[metric][j], float):
                    plt.bar(j, report[metric][j])
                    barplots = True
                else:
                    plt.plot(
                        report[metric][j]["x"],
                        report[metric][j]["y"],
                        label=report["header"][1 + j],
                    )
            if barplots:
                plt.xticks(list(range(len(report["header"][1:]))), report["header"][1:])
                plt.xticks(rotation=-xrotation, ha="right" if xrotation < 0 else "left")
            else:
                plt.legend()
            plt.title(metric)
            i += 1
    plt.tight_layout()
    if not hold:
        plt.show()
