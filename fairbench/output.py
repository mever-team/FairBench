from fairbench.fork import parallel, Fork
from matplotlib import pyplot as plt
import json


def _is_dict_of_dicts(report):
    return isinstance(report[next(iter(report))], dict)


def tojson(report):
    assert isinstance(report, Fork)
    report = report.branches
    data = dict()
    if not _is_dict_of_dicts(report):
        report = {"": report}
    data["header"] = ["Metric"]+[key for key in report]
    for value in report.values():
        for metric in value:
            if metric not in data:
                data[metric] = list()
            data[metric].append(float(f"{value[metric]}"))
    return json.dumps(data)


def describe(report: Fork, spacing: int = 15):
    assert isinstance(report, Fork)
    report = json.loads(tojson(report))
    ret = ""
    if report["header"]:
        ret += " ".join([entry.ljust(spacing) for entry in report["header"]])+"\n"
    for metric in report:
        if metric == "header":
            continue
        ret += " ".join([metric.ljust(spacing)]+[f"{entry:.3f}".ljust(spacing) for entry in report[metric]])+"\n"
    print(ret)


@parallel
def visualize(report: dict):
    for i, key in enumerate(report.keys()):
        plt.subplot(1, len(report), i + 1)
        plt.bar([key], [report[key]])
    plt.show()
