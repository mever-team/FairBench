def lfpr(alpha=0.9):
    import pygrank as pg

    return pg.LFPR(alpha=alpha, max_iters=1000, tol=1.0e-9)


def ppro(alpha=0.9):
    import pygrank as pg

    return pg.PageRank(alpha=alpha, max_iters=1000, tol=1.0e-9) >> pg.AdHocFairness("O")
