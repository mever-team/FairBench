import fairbench as fb
import numpy as np

y_true = np.array([10.2, 15.5, 8.1, 20.0, 12.3, 9.8, 18.4, 11.1])
y_pred = np.array([10.0, 15.0, 9.0, 19.0, 11.0, 9.5, 25.0, 10.5])
groups = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])

sensitive = fb.Dimensions(fb.categories @ groups)

report = fb.reports.pairwise(
    scores=y_pred, targets=y_true, sensitive=sensitive, score_bound="unbounded"
)

report.show(env=fb.export.Console(ansiplot=False))
