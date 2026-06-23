import fairbench as fb

experiments = {
    "flac utkface": lambda: fb.bench.vision.celeba(classifier="flac"),
    "badd utkface": lambda: fb.bench.vision.celeba(classifier="badd"),
}

settings = fb.Progress("settings")
for name, experiment in experiments.items():
    y, yhat, sensitive = experiment()
    sensitive = fb.Dimensions(fb.categories @ sensitive)
    report = fb.reports.pairwise(predictions=yhat, labels=y, sensitive=sensitive)
    settings.instance(name, report)
comparison = settings.build()

from ansiprint import AnsiTee
with AnsiTee.activate("ansi.html"):
    comparison.show(fb.export.ConsoleTable)
#
# settings = fb.Progress("settings")
# for name, experiment in experiments.items():
#     repetitions = fb.Progress("5 repetitions")
#     for repetition in range(5):
#         y, yhat, sensitive = experiment()
#         sensitive = fb.Dimensions(fb.categories @ sensitive)
#         report = fb.reports.pairwise(predictions=yhat, labels=y, sensitive=sensitive)
#         repetitions.instance(f"repetition {repetition}", report)
#
#     # get the average across repetitions
#     mean_report = repetitions.build().filter(fb.reduction.mean)
#     settings.instance(name, mean_report)
#
# comparison = settings.build()
# comparison["gm measures mean"].show(
#     env=fb.export.HtmlTable(sideways=True, transpose=True, legend=True)
# )
