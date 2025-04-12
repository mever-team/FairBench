from fairbench import v2 as fb
from sklearn import metrics
import numpy as np


def test_comparison():
    for i in range(1000):
        scores = np.random.rand(i)
        target = np.random.rand(i)
        sens = np.random.rand(i) > 0.3
        if np.random.rand() < 0.1:
            sens = None
        if np.random.rand() < 0.1 and i:
            sens = np.ones(i)
        yhat = scores > 0.5
        y = target > 0.5

        s = np.ones(i, dtype=bool) if sens is None else sens == 1

        # classification measures
        fba = {"predictions": yhat, "labels": y, "sensitive": sens}
        ska = {"y_pred": yhat[s], "y_true": y[s]}

        fb.measures.acc(**fba).testeq(metrics.accuracy_score(**ska))
        fb.measures.ppv(**fba).testeq(metrics.precision_score(**ska))
        fb.measures.tpr(**fba).testeq(metrics.recall_score(**ska))
        fb.measures.tnr(**fba).testeq(
            metrics.recall_score(y_pred=1 - yhat[s], y_true=1 - y[s])
        )
        fb.measures.f1(**fba).testeq(metrics.f1_score(**ska))
        fb.measures.pr(predictions=yhat, sensitive=sens).testeq(yhat[s].mean())
        fb.measures.tar(**fba).testeq((yhat * y)[s].mean())
        fb.measures.trr(**fba).testeq(((1 - yhat) * (1 - y))[s].mean())
        val = metrics.precision_score(**ska) / np.mean(y[s]) if np.mean(y[s]) > 0 else 0
        fb.measures.slift(**fba).testeq(val / (1 + val))
        fb.measures.nmcc(**fba).testeq((metrics.matthews_corrcoef(**ska) + 1) / 2)
        fb.measures.nkappa(**fba).testeq(
            (metrics.cohen_kappa_score(y[s], yhat[s]) + 1) / 2
        )

        # recommendation measures
        fba = {"scores": scores, "labels": y, "sensitive": sens}

        if s.sum() >= 3:
            # fb.measures.avgscore(scores=scores).testeq(np.mean(scores[s]))
            if len(np.unique(y[s])) > 1:
                fb.measures.auc(**fba).testeq(
                    metrics.roc_auc_score(y_true=y[s], y_score=scores[s])
                )
            fb.measures.ndcg(**fba).testeq(
                metrics.ndcg_score(y_true=[y[s]], y_score=[scores[s]])
            )
            fb.measures.topndcg(**fba).testeq(
                metrics.ndcg_score([y[s]], [scores[s]], k=3)
            )
        ranks = np.argsort(np.argsort(-scores[s]))
        relevant = np.where(y[s])[0]
        if len(relevant) > 0:
            rr = 1.0 / (1 + ranks[relevant].min())
            rr /= np.log2(len(ranks) + 1)
            fb.measures.nmrr(**fba).testeq(rr)
        else:
            fb.measures.nmrr(**fba).testeq(0)
