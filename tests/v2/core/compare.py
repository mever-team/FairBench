import math

from fairbench import v2 as fb
from sklearn import metrics as M
import numpy as np
from scipy.stats import spearmanr
import rbo


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

        fb.measures.acc(**fba).testeq(M.accuracy_score(**ska))
        fb.measures.ppv(**fba).testeq(M.precision_score(**ska))
        fb.measures.tpr(**fba).testeq(M.recall_score(**ska))
        fb.measures.tnr(**fba).testeq(
            M.recall_score(y_pred=1 - yhat[s], y_true=1 - y[s])
        )
        fb.measures.f1(**fba).testeq(M.f1_score(**ska))
        fb.measures.pr(predictions=yhat, sensitive=sens).testeq(yhat[s].mean())
        fb.measures.tar(**fba).testeq((yhat * y)[s].mean())
        fb.measures.trr(**fba).testeq(((1 - yhat) * (1 - y))[s].mean())
        val = M.precision_score(**ska) / np.mean(y[s]) if np.mean(y[s]) > 0 else 0
        fb.measures.lift(**fba).testeq(val)
        fb.measures.mcc(**fba).testeq(M.matthews_corrcoef(**ska))
        fb.measures.kappa(**fba).testeq(M.cohen_kappa_score(y[s], yhat[s]))

        # recommendation measures
        fba = {"scores": scores, "labels": y, "sensitive": sens}
        if s.sum() >= 3:
            # fb.measures.avgscore(scores=scores).testeq(np.mean(scores[s]))
            if len(np.unique(y[s])) > 1:
                fb.measures.auc(**fba).testeq(
                    M.roc_auc_score(y_true=y[s], y_score=scores[s])
                )
        ranks = np.argsort(np.argsort(-scores[s]))
        relevant = np.where(y[s])[0]
        if len(relevant) > 0:
            rr = 1.0 / (1 + ranks[relevant].min())
            rr /= np.log2(len(ranks) + 1)
            fb.measures.nmrr(**fba).testeq(rr)
        else:
            fb.measures.nmrr(**fba).testeq(0)

    # regression measures
    fba = {"scores": scores, "targets": target, "sensitive": sens}
    s = np.ones_like(scores, dtype=bool) if sens is None else sens == 1
    fb.measures.mabs(**fba).testeq(M.mean_absolute_error(target[s], scores[s]))
    fb.measures.mse(**fba).testeq(M.mean_squared_error(target[s], scores[s]))
    fb.measures.rmse(**fba).testeq(fb.measures.mse(**fba).float() ** 0.5)
    fb.measures.r2(**fba).testeq(M.r2_score(target[s], scores[s]))
    fb.measures.pinball(**fba).testeq(M.mean_pinball_loss(target[s], scores[s]))

    # ranking measures
    fba = {"scores": scores, "order": target, "sensitive": sens}
    if len(np.unique(scores[s])) > 1 and len(np.unique(target[s])) > 1:
        corr, _ = spearmanr(scores[s], target[s])
        fb.measures.spearman(**fba).testeq(corr)
    else:
        fb.measures.spearman(**fba).testeq(math.nan)
    """if s.sum() >= 3:
        fb.measures.ndcg(**fba).testeq(
            M.ndcg_score(
                y_true=target[s].reshape(1, -1), y_score=scores[s].reshape(1, -1)
            )
        )
        fb.measures.topndcg(**fba).testeq(
            M.ndcg_score(target[s].reshape(1, -1), scores[s].reshape(1, -1), k=3)
        )"""
    # fb.measures.rbo(**fba).testeq(rbo.RankingSimilarity(scores, target).rbo())
