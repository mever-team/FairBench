import fairbench as fb

if __name__ == "__main__":
    preds = [1, 1, 1, 0, 0, 1, 0]
    labels = [0, 1, 1, 0, 1, 0, 0]
    sensitive = fb.Fork(all=[1, 1, 1, 1, 1, 1, 1])

    report = fb.accreport(
        predictions=preds,
        labels=labels,
        sensitive=sensitive,
        metrics=[fb.metrics.fpr, fb.metrics.fnr, fb.metrics.positives, fb.metrics.tpr],
    )
    fb.describe(report)
