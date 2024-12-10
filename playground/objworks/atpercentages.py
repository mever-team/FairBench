from docutils.nodes import legend

from fairbench import v2 as fb
import fairbench as fb1
import numpy as np


comparison = fb.Progress("settings")
for test_size in np.arange(0.1, 1, 0.1):
    x, yhat, y = fb1.bench.tabular.bank(test_size=test_size)
    sensitive = fb1.categories @ x["marital"]

    report = fb.reports.pairwise(
        sensitive=sensitive,
        predictions=yhat,
        labels=y,
        measures=[fb.measures.trr, fb.measures.tar],
    )
    comparison.instance(f"test split {test_size:.3f}", report)

comparison = comparison.build()

# comparison.min.explain.show(fb.export.ConsoleTable(legend=False), depth=2)
comparison.explain.show(fb.export.ConsoleTable, depth=2)
