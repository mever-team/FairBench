import sys

import yaml
from fairbench.reports import Fork


def toyamlprimitives(report):
    if isinstance(report, Fork):
        report = report.branches()
    result = list()
    for metric, value in report.items():
        metric_dict = {
            'name': metric,
            'description': value.desc.desc,
            'results': str(value),
            'caveats': value.desc.caveats
        }
        result.append(metric_dict)
    return result


def toyaml(report):
    return yaml.dump({"Metrics": toyamlprimitives(report)})
