import math

from fairbench import ExplainableError
from fairbench.core.fork import Fork, Forklike
from fairbench.reports.accumulate import todict
import json
from fairbench.core.explanation.base import tofloat
from fairbench.core import ExplanationCurve


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
        if isinstance(value, ExplainableError) or isinstance(value, str):
            raise Exception(
                "Some entries in the report your are converting to json (either directly or as part of visualization) are strings or explainable errors."
            )
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


def describe(
    report: Fork,
    spacing: int = 15,
    show: bool = True,
    separator: str = " ",
    newline="\n",
):
    report = json.loads(tojson(report))
    ret = ""
    if report["header"]:
        ret += (
            separator.join([entry.ljust(spacing) for entry in report["header"]])
            + newline
        )
    for metric in report:
        if metric != "header":
            ret += (
                separator.join(
                    [metric.ljust(spacing)]
                    + [f"{entry:.3f}".ljust(spacing) for entry in report[metric]]
                )
                + newline
            )
    if show:
        print(ret)
    return ret


def text_visualize(report: Fork):
    report = json.loads(tojson(report))
    num_metrics = len([metric for metric in report if metric != "header"])
    i = 1
    for metric in report:
        if metric != "header":
            if num_metrics > 1:
                print(
                    "---------------------"
                    + (f"_{metric}_").center(10).replace(" ", "-").replace("_", " ")
                    + "---------------------"
                )
            max_value = 1
            for j, case in enumerate(report["header"][1:]):
                val = report[metric][j]
                if math.isnan(val):
                    continue
                val = abs(val)
                if val > max_value:
                    max_value = val
            for j, case in enumerate(report["header"][1:]):
                if isinstance(report[metric][j], float):
                    # plt.bar(j, report[metric][j])
                    if len(case) > 30:
                        case = case[:28] + ".."
                    if math.isnan(report[metric][j]):
                        continue
                    value = f"{float(report[metric][j]):.3f}"
                    print(
                        case.ljust(30)
                        + value.ljust(7)
                        + " | "
                        + "█" * int(report[metric][j] / max_value * 10)
                    )
                else:
                    raise Exception(
                        "fairbench.text_visualize does not yet support curve visualization"
                    )
            i += 1


def visualize(
    report: Fork, hold: bool = False, xrotation: int = None, legend: bool = True
):
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        print(
            "Interactive visualization dependencies are installed by `pip install fairbench[interactive]`."
            "\nRun `pip install matplotlib` to enable `fairbench.visualize`. For now, `fairbench.text_visualize` is used as a fallback."
        )
        return text_visualize(report)
    report = json.loads(tojson(report))
    num_metrics = len([metric for metric in report if metric != "header"])
    i = 1
    for metric in report:
        if metric != "header":
            if num_metrics > 1:
                plt.subplot(2, len(report) // 2, i)
            barplots = False
            if xrotation is None:
                xrotation = 90 if len((report["header"][1:])) > 5 else 0
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
            elif legend:
                plt.legend()
            plt.title(metric)
            i += 1
    if num_metrics > 1:
        plt.tight_layout()
    if not hold:
        plt.show()
