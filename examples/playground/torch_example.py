import fairbench as fb
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    scores = torch.Tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    preds = torch.Tensor([1, 1, 1, 0, 0, 1])
    labels = torch.Tensor([0, 1, 1, 0, 1, 0])

    gender = ["Man", "Man", "Woman", "Man", "Woman", "Man"]
    sensitives = fb.Fork(fb.categories @ gender)
    report = fb.unireport(
        scores=scores, predictions=preds, labels=labels, sensitive=sensitives, top=3
    )  # => crashes IndexError: index -3 is out of bounds for axis 0 with size 2

    # For example for ROC curves, try (for a multireport or uniport):
    fb.visualize(report.auc.maxbarea.explain.explain)
    # or to show the curve only:
    fb.visualize(report.auc.maxbarea.explain.explain.curve)
    # For curves of score distributions use:
    fb.visualize(report.avgscore.maxbarea.explain.explain.curve)

    # fb.visualize(report.auc.maxbarea.explain.explain)
    plt.savefig("roc.png")
