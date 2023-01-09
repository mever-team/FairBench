from fairbench.modal import multimodal, Modal
from matplotlib import pyplot as plt


def _is_dict_of_dicts(report):
    return isinstance(report[next(iter(report))], dict)


def describe(report: Modal, mode: str = None, spacing: int = 15):
    assert isinstance(report, Modal)
    report = report.modes
    ret = "--- " + mode + " ---\n" if mode else ""
    if not _is_dict_of_dicts(report):
        report = {"": report}
    else:
        ret += (
            (" " * spacing)
            + " "
            + " ".join(key.ljust(spacing) for key in report)
            + "\n"
        )
    rets = dict()
    for value in report.values():
        for metric in value:
            rets[metric] = (
                rets.get(metric, "") + f"{value[metric]:.3f}".ljust(spacing) + " "
            )
    for key, value in rets.items():
        ret += f"{key.ljust(spacing)} {value}\n"
    print(ret)


@multimodal
def visualize(report: dict):
    for i, key in enumerate(report.keys()):
        plt.subplot(1, len(report), i + 1)
        plt.bar([key], [report[key]])
    plt.show()
