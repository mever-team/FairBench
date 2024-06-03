import fairbench as fb
import torch

report = fb.multireport(
    predictions=torch.randint(2, (2,)),
    labels=torch.randint(2, (2,)),
    sensetive=fb.Fork(fb.categories @ ["m", "f"]),
    metrics=[fb.metrics.accuracy, fb.metrics.pr, fb.metrics.fpr, fb.metrics.fnr],
)
print(report)
stamps = fb.combine(
    fb.stamps.prule(report), fb.stamps.accuracy(report), fb.stamps.four_fifths(report)
)
print(fb.stamps.available())
print(stamps.symbols)
fb.modelcards.tohtml(stamps, file="model_card.html", show=True)
