import fairbench as fb
import pyfop as pfp
from sklearn.linear_model import LogisticRegression


def load():
    x = [[0, 0.1], [0.9, 0], [0, 0.1], [1.1, -0.1], [0.1, 0.1], [0.9, 0.1], [0.4, 0.5], [0.6, 0.3]]
    y = [0, 1, 0, 1, 0, 1, 1, 1]
    s = [0, 0, 0, 0, 1, 1, 1, 1]
    return x, y, s


x, y, s = load()
s2 = [0, 1, 1, 0, 0, 1, 0, 1]

x = fb.Modal(train=fb.array(x), test=fb.array(x))
y = fb.Modal(train=fb.array(y), test=fb.array(y))
sensitive = fb.Modal(train=fb.array(s), test=fb.array(s))

classifier = LogisticRegression()
classifier = fb.fit(classifier, x, y).train  # keep only training outcome
yhat = fb.predict(classifier, x)

# example of aggregating an objective
objective = fb.aggregate(test=fb.prule(yhat, sensitive), train=fb.accuracy(yhat, y))
yscores = fb.predict_proba(classifier, x)
yscores = fb.culep(yscores, y, sensitive, objective)

# example of generating a report under various notions of fairness for a data mode
print(fb.report(round(yscores), y=y, sensitive=sensitive).test())


#predictions = fb.culep(yscores, objective, sensitive)

#original_predictions = yscores()
#predictions = yscores.aspects(sample_weight=fb.skew(original_predictions, y, sensitive).train)
#print(predictions.train(culep_params=[0.5, -0.5, 1, 1]))
#print(predictions.train(culep_params=[0, 0, 0, 0]))

#print(predictions.train.get_input_context(culep_params=[0, 0, 0, 1]))
#print(predictions.train.get_input_context(culep_params=[1, 1, 0, 1]))


#yhat = fb.culep(yhat, y, objective)

#print(yhat.test(sensitive=sensitive))

#print(fb.report(yhat, y=y, sensitive=sensitive).test())
