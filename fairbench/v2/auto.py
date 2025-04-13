import numpy as np

from fairbench.v2.core import Descriptor, Value
from fairbench.v2.reports.adhoc import (
    reductions_vs_any,
    reductions_pairwise,
    all_measures,
    pairwise,
    vsall,
)

report_types = {
    "pairwise": (pairwise, reductions_pairwise, all_measures),
    "vsall": (vsall, reductions_vs_any, all_measures),
}


class MeasureBuilder:
    def __init__(self, parts):
        parts = set(parts)
        rep = [rep for rep in report_types if rep in parts]
        assert len(rep) == 1, (
            f"Quick measure builder requires one report type but found: {rep}"
            "Candidates: pairwise, vsall"
        )
        self.report, reductions, measures = report_types[rep[0]]
        self.reductions = [red for red in reductions if red.descriptor.name in parts]
        self.measures = [meas for meas in measures if meas.descriptor.name in parts]
        assert len(self.measures) == 1, (
            f"Quick measure builder requires one base measure but found: {self.measures}\n"
            f"Candidates for {rep} reports: {', '.join(meas.descriptor.name for meas in measures)}"
        )
        assert len(self.reductions) == 1, (
            f"Quick measure builder requires one reduction but found: {self.reductions}\n"
            f"Candidates for {rep} reports: {', '.join(red.descriptor.name for red in reductions)}"
        )
        self.name = (
            rep[0]
            + "_"
            + self.reductions[0].descriptor.name
            + "_"
            + self.measures[0].descriptor.name
        )

    def __call__(self, *args, **kwargs):
        ret: Value = (
            self.report(
                measures=self.measures, reductions=self.reductions, *args, **kwargs
            )
            | self.reductions[0]
            | self.measures[0]
        )
        desc = Descriptor(
            name=self.name,
            role="analysis " + ret.descriptor.role,
            details=(
                ret.descriptor.details
                + (
                    " of analysis that includes the whole population ('all') to compare against."
                    if self.report == vsall
                    else " of analysis that compares groups pairwise."
                )
            ),
            alias=ret.descriptor.alias,
            preferred_units=ret.descriptor.preferred_units,
        )
        desc.descriptor.details += (
            "# Caveats and recommendations "
            "\n • This is a generic list of caveats that apply to all measures."
            "\n • Non-quantitative criteria may also impact perceived fairness."
            "\n • Choose carefully the criteria on when measures are considered close to their ideal values."
            "\n • A single measure cannot decide whether a system is fair or biased without further investigation. "
            "It can at best indicate the absense of a particular bias. However, "
            "different measures are often at odds with each other, even when they have similar optima. "
            "\n • Consult with stakeholders to determine on which social and legal criteria systems should "
            "follow. This translates to choosing measures appropriate for the operating context."
            + "\n# Distribution"
        )
        return ret.rebase(desc)


class QuickMeasures:
    def __init__(self):
        pass

    def __getattr__(self, item):
        if item in dir(self):
            return AttributeError
        return MeasureBuilder(str(item).split("_"))

    def help(self):
        print(
            "Showing all fairness measures that can be computed.\n"
            "These are dynamically created from building blocks.\n"
            "Create reports to capture many of those.\n"
        )
        for item in self:
            print("fairbench.quick." + item)

    def __iter__(self):
        import fairbench as fb

        scores = np.random.rand(1000)
        target = np.random.rand(1000)
        sensitive = np.random.rand(1000)
        kw = {
            "scores": scores,
            "targets": target,
            "predictions": scores > 0.5,
            "labels": target > 0.5,
            "order": target,
            "sensitive": fb.Dimensions(fb.fuzzy @ sensitive),
        }
        for rep in report_types:
            for red in report_types[rep][1]:
                for meas in report_types[rep][2]:
                    item = rep + "_" + red.descriptor.name + "_" + meas.descriptor.name
                    value = getattr(self, item)(**kw)
                    if value.exists():
                        yield item


quick = QuickMeasures()
