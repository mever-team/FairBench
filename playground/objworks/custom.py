import fairbench as fb

def custom_report(sensitive, predictions, labels):
    prule = fb.quick.pairwise_maxrel_pr(sensitive=sensitive, predictions=predictions, labels=labels).rebase("prule")
    prule.value = fb.core.TargetedNumber(1-float(prule), target=1.0)
    return fb.core.Value(
        descriptor=fb.core.Descriptor(name="my custom report", role="assessment"),
        depends=[
            fb.quick.pairwise_wmean_acc(sensitive=sensitive, predictions=predictions, labels=labels).rebase("acc"),
            fb.quick.pairwise_min_acc(sensitive=sensitive, predictions=predictions, labels=labels).rebase("min acc"),
            prule,
            fb.quick.pairwise_maxdiff_tpr(sensitive=sensitive, predictions=predictions, labels=labels).rebase("|Δtpr|"),
            fb.quick.pairwise_maxdiff_tnr(sensitive=sensitive, predictions=predictions, labels=labels).rebase("|Δtnr|"),
        ]
    )

sensitive = fb.Dimensions(men=[1, 1, 0, 0, 0], women=[0, 0, 1, 1, 1])
labels=[1, 0, 0, 1, 0]
predictionsA=[1, 0, 1, 0, 0]
predictionsB=[1, 0, 1, 0, 1]
comparisons = fb.Progress("comparisons")
comparisons["system A"] = custom_report(sensitive, predictionsA, labels)
comparisons["system B"] = custom_report(sensitive, predictionsB, labels)
from ansiprint import AnsiTee
with AnsiTee.activate("ansi.html"):
    comparisons.build().show(env=fb.export.ConsoleTable(transpose=True))
