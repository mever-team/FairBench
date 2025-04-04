from fairbench import v2 as fb
import fairbench as deprecated

x, y, yhat = deprecated.bench.tabular.bank()

sensitive = fb.Dimensions(
    {"marital": fb.categories @ x["marital"], "age": fb.fuzzy @ x["age"]}
)
print(sensitive.branches)
report = fb.reports.pairwise(
    sensitive=sensitive,
    predictions={"true": yhat, "false": 1 - yhat},
    labels={"true": y, "false": 1 - y},
)

report.filter(fb.investigate.Stamps, fb.investigate.Worst).show(
    env=fb.export.Html, depth=2
)
print(report.filter(fb.investigate.Stamps))

report.filter(fb.investigate.Stamps).show()


"""
print(comparison)

value = (comparison
          |fb.reduction.minimum
          |fb.measures.pr)

print(value)

for v in value.flatten():
    print(v)
"""

# print(json.dumps((report_over_time|pr|minimum).serialize(), indent="  "))
