import math

from fairbench.v1.core import Fork, ExplainableError
from fairbench.v1.reports.accumulate import todict
import json
from fairbench.v1.core.explanation.base import tofloat
from fairbench.v1.core import ExplanationCurve
from typing import Optional
import re


def _is_fork_of_dicts(report):
    return isinstance(report[next(iter(report))], dict)


def tojson(report: Fork):
    if isinstance(report, dict):  # includes DotDict
        report = todict(**report)
    if isinstance(report, dict):  # if it's still a DotDict
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
    spacing: int = 20,
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


def text_visualize(report: Fork, show: bool = True, save: Optional[str] = None):
    import ansiplot

    report = json.loads(tojson(report))
    num_metrics = len([metric for metric in report if metric != "header"])
    i = 1
    ret = ""
    for metric in report:
        if metric != "header":
            if num_metrics > 1:
                ret += "---------------------"
                ret += (f"_{metric}_").center(10).replace(" ", "-").replace("_", " ")
                ret += "---------------------\n"
            else:
                ret += "-" * 50 + "\n"
            max_value = 1
            for j, case in enumerate(report["header"][1:]):
                val = report[metric][j]
                if isinstance(val, dict) or math.isnan(val):
                    continue
                val = abs(val)
                if val > max_value:
                    max_value = val
            plotter = None
            for j, case in enumerate(report["header"][1:]):
                value = report[metric][j]
                if isinstance(value, float):
                    # plt.bar(j, report[metric][j])
                    if len(case) > 30:
                        case = case[:28] + ".."
                    if math.isnan(report[metric][j]):
                        continue
                    progress = int(value / max_value * 10)
                    value = f"{float(value):.3f}"
                    ret += (
                        case.ljust(30) + value.ljust(7) + " | " + "█" * progress + "\n"
                    )
                else:
                    if plotter is None:
                        plotter = ansiplot.Scaled(60, 10)
                    plotter.plot(value["x"], value["y"], title=case)
            if plotter is not None:
                ret += "\n" + plotter.text() + "\n"
            i += 1
    if show:
        print(ret)
    if save is not None:
        save = f"{save}.txt"
        with open(save, "w") as file:
            file.write(re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", ret))
        print(f"Plot saved to: {save}")


def visualize(
    report: Fork,
    show: bool = True,
    xrotation: int = None,
    legend: bool = True,
    save: Optional[str] = None,
):
    fallback = False
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        fallback = True
    if fallback:
        print(
            "Interactive visualization dependencies are installed by `pip install fairbench[interactive]`."
            "\nFor now, `fairbench.text_visualize` is used as a fallback."
        )
        text_visualize(report, show=show, save=save)
        return
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
    if num_metrics > 1 and xrotation != 0:
        plt.tight_layout()
    if show:
        plt.show()
    if save is not None:
        save = f"{save}.png"
        plt.savefig(save)
        print(f"Plot saved to: {save}")
