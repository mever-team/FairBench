from fairbench.forks.fork import Fork
from matplotlib import pyplot as plt
import json
from fairbench.forks.explanation import tofloat


def _is_dict_of_dicts(report):
    return isinstance(report[next(iter(report))], dict)


def tojson(report: Fork):
    assert isinstance(report, Fork)
    report = report.branches()
    data = dict()
    if not _is_dict_of_dicts(report):
        report = {k: {"": v} for k, v in report.items()}
    data["header"] = ["Metric"] + [key for key in report]
    for value in report.values():
        for metric in value:
            if metric not in data:
                data[metric] = list()
            data[metric].append(tofloat(value[metric]))
    return json.dumps(data)


def describe(report: Fork, spacing: int = 15):
    assert isinstance(report, Fork)
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


def visualize(report: Fork):
    assert isinstance(report, Fork)
    report = json.loads(tojson(report))

    i = 1
    for metric in report:
        if metric != "header":
            plt.subplot(2, len(report) // 2, i)
            for j, case in enumerate(report["header"][1:]):
                plt.bar(j, report[metric][j])
            plt.xticks(list(range(len(report["header"][1:]))), report["header"][1:])
            plt.title(metric)
            i += 1
    plt.tight_layout()
    plt.show()
