import fairbench as fb

manual_pr_values = fb.core.report_from_dims(fb.v1.Fork({"male": 0.3, "female": 0.4}))

manual_y = [1, 0, 1, 1, 1, 0]
manual_yhat = [1, 0, 1, 0, 1, 0]
manual_sensitive = fb.Dimensions(
    fb.categories @ ["male", "male", "male", "female", "female", "female"]
)
manual_report = fb.reports.pairwise(
    labels=manual_y, predictions=manual_yhat, sensitive=manual_sensitive
)

print(manual_pr_values)
print(manual_report.min.pr.details)

manual_report.min.pr.details.filter(
    fb.investigate.Amplification(base=manual_pr_values, default_to_zero_targets=True)
).show(depth=2)
