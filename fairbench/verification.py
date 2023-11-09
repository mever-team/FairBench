from fairbench.core import Fork, Forklike, Explainable, ExplainableError
from typing import Iterable


def _check_equals(fork1, fork2):
    assert isinstance(fork1, Fork) == isinstance(fork2, Fork)
    assert isinstance(fork1, dict) == isinstance(fork2, dict)
    if isinstance(fork1, Fork):
        _check_equals(fork1.branches(), fork2.branches())
    elif isinstance(fork1, dict):
        assert len(set(fork1.keys()) - set(fork2.keys())) == 0
        assert len(set(fork2.keys()) - set(fork1.keys())) == 0
        for k in fork1.keys():
            _check_equals(fork1[k], fork2[k])
    else:
        assert fork1 == fork2


class Stamp:
    def __init__(
        self,
        name,
        fields: Iterable,
        minimum=None,
        maximum=None,
        desc="Computed with FairBench.",
        caveats=[
            "Different fairness measures capture different concerns and may be at odds with each other."
        ],
    ):
        assert isinstance(fields, Iterable)
        if not isinstance(list(fields)[0], Iterable):
            fields = [fields]
        if isinstance(caveats, str):
            caveats = [caveats]
        fields = [
            field.split(".") if isinstance(field, str) else field for field in fields
        ]
        self.name = name
        self._fields = fields
        self.minimum = minimum
        self.maximum = maximum
        self.desc = desc
        self.caveats = caveats

    def __call__(self, report):
        rets = [self.__call_once(fields, report) for fields in self._fields]
        result = None
        for ret in rets:
            if not isinstance(ret, ExplainableError):
                if result is None or isinstance(ret, Explainable):
                    result = ret
                # _check_equals(result, ret)
        if result is None:
            result = ExplainableError(
                f"Report does not contain any of {', '.join('.'.join(fields) for fields in self._fields)}"
            )
        result.desc = self
        return Forklike({self.name: result})

    def __call_once(self, fields, report):
        assert isinstance(report, Fork)
        try:
            for field in fields:
                report = getattr(report, field)
        except AttributeError:
            return ExplainableError(f"Report does not contain {'.'.join(fields)}")
        original_report = report
        if self.minimum is not None and self.maximum is not None:
            report = (report >= self.minimum) & (report <= self.maximum)
        elif self.maximum is not None:
            report = report <= self.maximum
        elif self.minimum is not None:
            report = report >= self.minimum
        else:
            original_report = report.explain
        if isinstance(report, Fork):
            return Fork(
                {
                    k: v
                    if isinstance(v, Explainable) or isinstance(v, ExplainableError)
                    else Explainable(v, explain=original_report, units=bool)
                    for k, v in report.branches().items()
                }
            )
        return (
            report
            if isinstance(report, Explainable) or isinstance(report, ExplainableError)
            else Explainable(report, explain=original_report, units=bool)
        )


import requests
import yaml

class StampSpecs:
    def __init__(self, path="https://raw.githubusercontent.com/mever-team/FairBench/main/stamps/common.yaml"):
        self._stamps = dict()
        self._path = path
        self._resources = None

    def source(self, path):
        self._path = path
        self._resources = None

    def clear(self):
        self._resources = None
        self._path = None

    def __getattribute__(self, attr):
        if attr in ["_resources", "_stamps", "_path"]:
            return object.__getattribute__(self, attr)
        if self._resources is None and self._path is None:
            response = requests.get(self._path)
            if response.status_code == 200:
                self._resources = yaml.safe_load(response.text)
            else:
                raise Exception(f"Failed to download YAML file from {self._path}. Status code: {response.status_code}")

        if attr in self._stamps:
            return self._stamps[attr]
        resource = self._resources[attr]
        ret = Stamp(resource["title"],
                    resource["alias"],
                    minimum=resource.get("minimum", None),
                    maximum=resource.get("maximum", None),
                    desc=resource.get("description"),
                    caveats=resource.get("caveats"))
        self._stamps[attr] = ret
        return ret

stamps = StampSpecs()