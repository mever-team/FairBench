import yaml
from fairbench.v1.reports import Fork


def toyamlprimitives(report):
    if isinstance(report, Fork):
        report = report.branches()
    result = list()
    for metric, value in report.items():
        stamp = value.stamp
        symbols = stamp.symbols
        description = stamp.desc
        caveats = stamp.caveats
        value = str(value)
        if value == "True" and stamp.caveats_accept is not None:
            caveats = caveats + stamp.caveats_accept
        if value == "False" and stamp.caveats_reject is not None:
            caveats = caveats + stamp.caveats_reject
        if symbols:
            for symbol, replace in symbols.items():
                description = description.replace("{" + symbol + "}", replace)
                caveats = [
                    caveat.replace("{" + symbol + "}", replace) for caveat in caveats
                ]
        metric_dict = {
            "name": metric,
            "description": description,
            "results": value,
            "caveats": caveats,
        }
        result.append(metric_dict)
    return result


def toyaml(report):
    return yaml.dump({"Metrics": toyamlprimitives(report)})
