import fairbench as fb

sensitive = fb.Dimensions(men=[1, 1, 0, 0, 0], women=[0, 0, 1, 1, 1])
report = fb.reports.conflate(
    predictions=[1, 0, 1, 0, 0], labels=[1, 0, 0, 1, 0], sensitive=sensitive
)

from ansiprint import AnsiTee

with AnsiTee.activate("ansi.html"):
    report["men conflate"].maxdiff.show(env=fb.export.ConsoleTable)
