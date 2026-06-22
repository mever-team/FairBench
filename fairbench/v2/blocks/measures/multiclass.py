from fairbench.v2 import core as c
from fairbench.v2.blocks.quantities import quantities
import numpy as np


@c.measure("the multiclass accuracy")
def wmacc(multipredictions, multilabels, sensitive=None):
    multipredictions = np.array(
        multipredictions
    )  # shape (N,) with integer class indices
    multilabels = np.array(multilabels)  # shape (N,) with integer class indices
    sensitive = (
        np.ones(len(multipredictions)) if sensitive is None else np.array(sensitive)
    )
    correct = (multipredictions == multilabels).astype(float)
    samples = sensitive.sum()
    value = 0 if samples == 0 else (correct * sensitive).sum() / samples
    return c.Value(
        c.TargetedNumber(value, 1),
        depends=[quantities.samples(samples)],
    )


@c.measure("the arithmetic mean of per-class accuracies")
def amacc(multipredictions, multilabels, sensitive=None):
    multipredictions = np.array(multipredictions)
    multilabels = np.array(multilabels)
    sensitive = (
        np.ones(len(multipredictions)) if sensitive is None else np.array(sensitive)
    )
    classes = np.unique(np.unique(multilabels) + np.unique(multipredictions))
    per_class_accs = []
    for cls in classes:
        mask = (multilabels == cls).astype(float) * sensitive
        cls_samples = mask.sum()
        if cls_samples == 0:
            continue
        correct = (multipredictions == multilabels).astype(float)
        cls_acc = (correct * mask).sum() / cls_samples
        per_class_accs.append(cls_acc)
    value = 0 if len(per_class_accs) == 0 else float(np.mean(per_class_accs))
    samples = sensitive.sum()
    return c.Value(
        c.TargetedNumber(value, 1),
        depends=[quantities.samples(samples)],
    )


@c.measure("the geometric mean of per-class accuracies")
def gmacc(multipredictions, multilabels, sensitive=None):
    multipredictions = np.array(multipredictions)
    multilabels = np.array(multilabels)
    sensitive = (
        np.ones(len(multipredictions)) if sensitive is None else np.array(sensitive)
    )
    classes = np.unique(np.unique(multilabels) + np.unique(multipredictions))
    per_class_accs = []
    for cls in classes:
        mask = (multilabels == cls).astype(float) * sensitive
        cls_samples = mask.sum()
        if cls_samples == 0:
            continue
        correct = (multipredictions == multilabels).astype(float)
        cls_acc = (correct * mask).sum() / cls_samples
        per_class_accs.append(cls_acc)
    value = (
        0
        if len(per_class_accs) == 0 or any(a == 0 for a in per_class_accs)
        else float(np.prod(per_class_accs) ** (1.0 / len(per_class_accs)))
    )
    samples = sensitive.sum()
    return c.Value(
        c.TargetedNumber(value, 1),
        depends=[quantities.samples(samples)],
    )


@c.measure(
    "the arithmetic mean of per-class true positive rates/recalls/sensitivities/hit rates"
)
def amtpr(multipredictions, multilabels, sensitive=None):
    multipredictions = np.array(multipredictions)
    multilabels = np.array(multilabels)
    sensitive = (
        np.ones(len(multipredictions)) if sensitive is None else np.array(sensitive)
    )
    classes = np.unique(np.unique(multilabels) + np.unique(multipredictions))
    per_class = []
    for cls in classes:
        pos_mask = (multilabels == cls).astype(float) * sensitive
        ap = pos_mask.sum()
        if ap == 0:
            continue
        tp = ((multipredictions == cls).astype(float) * pos_mask).sum()
        per_class.append(tp / ap)
    value = 0 if len(per_class) == 0 else float(np.mean(per_class))
    return c.Value(
        c.TargetedNumber(value, 1),
        depends=[quantities.samples(sensitive.sum())],
    )


@c.measure(
    "the geometric mean of per-class true positive rates/recalls/sensitivities/hit rates"
)
def gmtpr(multipredictions, multilabels, sensitive=None):
    multipredictions = np.array(multipredictions)
    multilabels = np.array(multilabels)
    sensitive = (
        np.ones(len(multipredictions)) if sensitive is None else np.array(sensitive)
    )
    classes = np.unique(np.unique(multilabels) + np.unique(multipredictions))
    per_class = []
    for cls in classes:
        pos_mask = (multilabels == cls).astype(float) * sensitive
        ap = pos_mask.sum()
        if ap == 0:
            continue
        tp = ((multipredictions == cls).astype(float) * pos_mask).sum()
        per_class.append(tp / ap)
    value = (
        0
        if len(per_class) == 0 or any(v == 0 for v in per_class)
        else float(np.prod(per_class) ** (1.0 / len(per_class)))
    )
    return c.Value(
        c.TargetedNumber(value, 1),
        depends=[quantities.samples(sensitive.sum())],
    )


@c.measure("the arithmetic mean of per-class positive predictive values/precisions")
def amppv(multipredictions, multilabels, sensitive=None):
    multipredictions = np.array(multipredictions)
    multilabels = np.array(multilabels)
    sensitive = (
        np.ones(len(multipredictions)) if sensitive is None else np.array(sensitive)
    )
    classes = np.unique(np.unique(multilabels) + np.unique(multipredictions))
    per_class = []
    for cls in classes:
        pred_mask = (multipredictions == cls).astype(float) * sensitive
        pp = pred_mask.sum()  # all predicted as cls (within sensitive group)
        if pp == 0:
            continue
        tp = ((multilabels == cls).astype(float) * pred_mask).sum()
        per_class.append(tp / pp)
    value = 0 if len(per_class) == 0 else float(np.mean(per_class))
    return c.Value(
        c.TargetedNumber(value, 1),
        depends=[quantities.samples(sensitive.sum())],
    )


@c.measure("the geometric mean of per-class positive predictive values/precisions")
def gmppv(multipredictions, multilabels, sensitive=None):
    multipredictions = np.array(multipredictions)
    multilabels = np.array(multilabels)
    sensitive = (
        np.ones(len(multipredictions)) if sensitive is None else np.array(sensitive)
    )
    classes = np.unique(np.unique(multilabels) + np.unique(multipredictions))
    per_class = []
    for cls in classes:
        pred_mask = (multipredictions == cls).astype(float) * sensitive
        pp = pred_mask.sum()
        if pp == 0:
            continue
        tp = ((multilabels == cls).astype(float) * pred_mask).sum()
        per_class.append(tp / pp)
    value = (
        0
        if len(per_class) == 0 or any(v == 0 for v in per_class)
        else float(np.prod(per_class) ** (1.0 / len(per_class)))
    )
    return c.Value(
        c.TargetedNumber(value, 1),
        depends=[quantities.samples(sensitive.sum())],
    )
