from fairbench.experimental import core_v2 as c


class quantities:
    samples = c.Descriptor("samples", "count")
    positives = c.Descriptor("positives", "count", "positive predictions")
    negatives = c.Descriptor("negatives", "count", "negative predictions")
    tp = c.Descriptor("tp", "count", "true positive predictions")
    tn = c.Descriptor("tn", "count", "true negative predictions")
    ap = c.Descriptor("ap", "count", "actual positive labels")
    an = c.Descriptor("an", "count", "actual negative labels")
    freedom = c.Descriptor("freedom", "parameter", "degrees of freedom")
    slope = c.Descriptor("slope", "parameter", "slope of the pinball deviation")
