import fairbench as fb
import torch
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    preds = torch.Tensor([1,1,1,0,0,1])
    labels = torch.Tensor([0,1,1,0,1,0])
    sensitive = fb.Fork(all=[1,1,1,1,1,1])

    report = fb.accreport(predictions=preds, labels=labels, sensitive=sensitive,
                          metrics=[fb.metrics.fpr, fb.metrics.fnr, fb.metrics.positives])
    fb.describe(report)
    tn, fp, fn, tp = confusion_matrix(labels, preds.tolist()).ravel()
    print(fp, fp+tn)
    print(report.fpr.explain.false_positives,report.fpr.explain.negatives)
    print('expected fpr: {}, fnr : {}'.format((fp / (fp + tn)) * 100, (fn / (fn+tp)) * 100))