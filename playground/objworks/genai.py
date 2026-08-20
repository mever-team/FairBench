import fairbench as fb

baseline_x, baseline_y, baseline_yhat = fb.bench.tabular.compas(test_size=0.2)
baseline_sensitive = fb.Dimensions(
    fb.categories @ baseline_x["sex"], fb.categories @ baseline_x["race"]
)
baseline_sensitive = baseline_sensitive.intersectional().strict()

x, y, yhat = fb.bench.tabular.compas(test_size=0.5)
sensitive = fb.Dimensions(fb.categories @ x["sex"], fb.categories @ x["race"])
sensitive = sensitive.intersectional().strict()


# reference: https://prasannavijay.github.io/fairbench/whitepaper/

# OUTPUT DIVERSITY ENTROPY
# blocks.measures.ode -> included in reports

# Harm Severity Index -> bias of a harm predictor

# BIAS AMPLIFICATIONS
print(sensitive.sum())
baseline_report = fb.reports.pairwise(
    multipredictions=baseline_yhat, multilabels=baseline_y, sensitive=baseline_sensitive
)
report = fb.reports.pairwise(multipredictions=yhat, multilabels=y, sensitive=sensitive)
report = report.filter(
    fb.investigate.Amplification(base=baseline_report), fb.investigate.IsBias
)
report.show(fb.export.Html, depth=2)


# REPRESENTATION SKEW
target_pr_values = fb.core.report_from_dims(sensitive.sum() / sensitive.shape[0])
print(target_pr_values)
report = fb.reports.pairwise(predictions=yhat, sensitive=sensitive).min.pr
print(report.details)
report = report.details.filter(
    fb.investigate.SetTargets(targets=target_pr_values), fb.investigate.SkewIndex
)
report.show(env=fb.export.Html)
