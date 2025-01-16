from fairbench import v2 as fb

experiments = {
    "flac utkface": lambda: fb.bench.vision.utkface(classifier="flac"),
    "badd utkface": lambda: fb.bench.vision.utkface(classifier="badd"),
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

print("================================================")
comparison.show(fb.export.ConsoleTable)
print("================================================")
comparison.explain.show(env=fb.export.ConsoleTable(sideways=False))
print("================================================")
comparison.acc.explain["maxdiff explain mean"].show(env=fb.export.Console)
print("================================================")
filter = fb.investigate.DeviationsOver(0.1)
env = fb.export.ConsoleTable(
    sideways=False
)  # not sideways because the environment complains about different rows
comparison.filter(filter).show(env=env)
