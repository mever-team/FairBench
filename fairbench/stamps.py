from fairbench.forks import Fork, Forklike, Explainable, ExplainableError
from typing import Iterable


def _check_equals(fork1, fork2):
    assert isinstance(fork1, Fork) == isinstance(fork2, Fork)
    assert isinstance(fork1, dict) == isinstance(fork2, dict)
    if isinstance(fork1, Fork):
        _check_equals(fork1.branches(), fork2.branches())
    elif isinstance(fork1, dict):
        assert len(set(fork1.keys())-set(fork2.keys())) == 0
        assert len(set(fork2.keys())-set(fork1.keys())) == 0
        for k in fork1.keys():
            _check_equals(fork1[k], fork2[k])
    else:
        assert fork1 == fork2


class Stamp:
    def __init__(self, name, fields: Iterable, minimum=None, maximum=None,
                 desc="Computed with FairBench.",
                 caveats=["Different fairness measures capture different concerns and may be at odds with each other."]):
        assert isinstance(fields, Iterable)
        if not isinstance(list(fields)[0], Iterable):
            fields = [fields]
        if isinstance(caveats, str):
            caveats = [caveats]
        fields = [field.split(".") if isinstance(field, str) else field for field in fields]
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
                #_check_equals(result, ret)
        if result is None:
            result = ExplainableError(f"Report does not contain any of {', '.join('.'.join(fields) for fields in self._fields)}")
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
            report = (report <= self.maximum)
        elif self.minimum is not None:
            report = (report >= self.minimum)
        else:
            original_report = report.explain
        if isinstance(report, Fork):
            return Fork({k: v if isinstance(v, Explainable) or isinstance(v, ExplainableError) else Explainable(v, explain=original_report, units=bool) for k, v in report.branches().items()})
        return report if isinstance(report, Explainable) or isinstance(report, ExplainableError) else Explainable(report, explain=original_report, units=bool)


three_fourths = Stamp("3/4ths ratio", ("minratio.pr", "prule"), minimum=0.8,
                      desc="Checks whether the fraction of positive predictions for each protected group "
                           "is at worst 3/4ths that of any other group.",
                      caveats=["Disparate impact may not always be an appropriate fairness consideration.",
                               "Consider input from affected stakeholders to determine whether "
                               "the 3/4ths ratio is an appropriate fairness criterion."])
eighty_rule = Stamp("80% rule", ("minratio.pr", "prule"), minimum=0.8,
                    desc="Checks whether the fraction of positive predictions for each protected group "
                         "is at worst 80% that of any other group.",
                    caveats=["Disparate impact may not always be an appropriate fairness consideration.",
                             "Consider input from affected stakeholders to determine whether "
                             "the 80% rule is an appropriate fairness criterion."])
accuracy = Stamp("worst accuracy", ("min.accuracy", "accuracy"),
                 desc="Computes the worst performance among protected groups; this is the minimum "
                      "benefit the system brings to any group.",
                 caveats=["The worst accuracy is a lower bound but not an estimation of overall accuracy. "
                          "There may be different distributions of benefits that could be protected."])
prule = Stamp("prule", ("minratio.pr", "prule"),
              desc="Compares the fraction of positive predictions between protected groups. "
                   "The worst ratio between the groups is reported, so that value of 0 indicates "
                   "indicates disparate impact, and value of 1 disparate impact mitigation.",
              caveats=["Disparate impact may not always be an appropriate fairness consideration."])
