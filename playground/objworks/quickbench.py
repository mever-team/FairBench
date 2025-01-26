from fairbench import v2 as fb

experiments = {
    "flac utkface": lambda: fb.bench.vision.utkface(classifier="flac"),
    "badd utkface": lambda: fb.bench.vision.utkface(classifier="badd"),
    "flac celeba": lambda: fb.bench.vision.celeba(classifier="flac"),
    "mavias celeba": lambda: fb.bench.vision.celeba(classifier="mavias"),
    "badd waterbirds": lambda: fb.bench.vision.waterbirds(classifier="badd"),
    "mavias waterbirds": lambda: fb.bench.vision.waterbirds(classifier="mavias"),
}

settings = fb.Progress("settings")
for name, experiment in experiments.items():
    repetitions = fb.Progress("5 repetitions")
    for repetition in range(5):
        y, yhat, sensitive = experiment()
        sensitive = fb.Dimensions(fb.categories @ sensitive)
        report = fb.reports.pairwise(predictions=yhat, labels=y, sensitive=sensitive)
        repetitions.instance(f"repetition {repetition}", report)
    mean_report = repetitions.build().filter(fb.reduction.mean)
    settings.instance(name, mean_report)

comparison = settings.build()

comparison["maxdiff explain mean"].explain.show()
# print("================================================")
# comparison.show(fb.export.ConsoleTable)
# print("================================================")
# comparison.explain.show(env=fb.export.ConsoleTable(sideways=False))
# print("================================================")
# comparison.acc.explain["maxdiff explain mean"].show(env=fb.export.Console)
# print("================================================")
# comparison.show(env=fb.export.ConsoleTable(sideways=False))
