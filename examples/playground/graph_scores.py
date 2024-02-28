import pygrank as pg
import fairbench as fb

# load data and set sensitive attribute
_, graph, communities = next(pg.load_datasets_multiple_communities(["highschool"]))
train, test = pg.split(pg.to_signal(graph, communities[0]), 0.5)
sensitive_signal = pg.to_signal(graph, communities[1])
labels = test.filter(exclude=train)
sensitive = fb.Fork(gender=fb.categories @ sensitive_signal.filter(exclude=train))

# create report for pagerank
algorithm = pg.PageRank(alpha=0.85)
scores = algorithm(train).filter(exclude=train)
report = fb.multireport(labels=labels, scores=scores, sensitive=sensitive, top=50)

# create report for locally fair pagerank
fair_algorithm = pg.LFPR(alpha=0.85, redistributor="original")
fair_scores = fair_algorithm(train, sensitive=sensitive_signal).filter(exclude=train)
fair_report = fb.multireport(labels=labels, scores=fair_scores, sensitive=sensitive, top=50)

# combine both reports into one and get the auc perspective
fork = fb.Fork(ppr=report, lfpr=fair_report)
#   fb.interactive(fork)
#fb.describe(fork.phi)
#print(fork.phi)

value = fb.areduce(fb.avghr(labels=labels, scores=fair_scores, sensitive=sensitive, top=50),
          reducer=fb.reducers.max,
          expand=fb.expanders.bdcg
        )
print(value.explain)