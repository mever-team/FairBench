from fairbench.v2 import core as c


class quantities:
    samples = c.Descriptor("samples", "count", "the sample count")
    unique = c.Descriptor("unique", "count", "the number of unique recommendation")
    positives = c.Descriptor("positives", "count", "the positive predictions")
    negatives = c.Descriptor("negatives", "count", "the negative predictions")
    tp = c.Descriptor("tp", "count", "the true positive predictions")
    tn = c.Descriptor("tn", "count", "the true negative predictions")
    ap = c.Descriptor("ap", "count", "the actual positive labels")
    an = c.Descriptor("an", "count", "the actual negative labels")
    freedom = c.Descriptor("freedom", "parameter", "the degrees of freedom")
    slope = c.Descriptor("slope", "parameter", "the slope of pinball deviation")
    distribution = c.Descriptor(
        "distribution",
        "curve",
        "the cumulative density function of the score distribution (numerically robust compared to non-cumulative)",
    )
    itemloss = c.Descriptor(
        "itemloss",
        "curve",
        "the loss exhibited by items at each position (the position is normalized)",
    )
    roc = c.Descriptor("roc", "curve", "the receiver operating characteristics curve")
    top = c.Descriptor("top", "count", "the number of top scores considered")
    precision = c.Descriptor("precision", "metric", "the precision score")
    repr = c.Descriptor("repr", "metric", "the representation in top samples")
