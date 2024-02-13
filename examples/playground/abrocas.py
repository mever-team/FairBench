import numpy as np
import fairbench as fb

# test, y, yhat = fb.demos.adult(predict="probabilities")
# s = fb.Fork(fb.categories @ test[8])  # test[8] is a pandas column with race values

sk = list()
a0 = list()
a1 = list()
a2 = list()
a3 = list()


for skew in np.arange(1, 2.01, 0.01):
    samples = 1000
    minority = 0.5
    positive_rate = 0.5
    bias = 0.5
    np.random.seed(6)  # 3 is default for experiment results
    skew = skew**5

    def f1(y):
        x = np.random.uniform(0, 1)
        if y == 1:
            return x  # 1-x**skew#x#1-x**skew
        return x**skew
        # return 1-f1(1-y)

    def f2(y):
        x = np.random.uniform(0, 1)
        if y == 1:
            return 1 - x**skew
        return x**skew
        # return 1-f2(1-y)

    sensitive = np.random.choice([0, 1], size=samples, p=[1 - minority, minority])
    labels = np.array(
        [
            np.random.choice(
                [0, 1],
                p=[1 - positive_rate, positive_rate]
                if s == 1
                else [1 - positive_rate * (1 - bias), (1 - bias) * positive_rate],
            )
            for s in sensitive
        ]
    )
    scores = np.array([f1(y) if s == 1 else f2(y) for s, y in zip(sensitive, labels)])

    """
    # sanity check about equal curves
    scores = np.concatenate([scores, scores])
    sensitive = np.concatenate([sensitive*0+1, sensitive*0])
    labels = np.concatenate([labels, labels])
    """

    aucs = fb.metrics.auc(
        scores=scores,
        labels=labels,
        sensitive=fb.Fork({"protected": sensitive, "non-protected": 1 - sensitive}),
    )

    def biimpl(y1, y2):
        numerator = np.maximum(y1, y2) - np.minimum(y1, y2)
        denominator = np.maximum(y1, y2)
        with np.errstate(invalid="ignore"):
            result = np.divide(numerator, denominator)
            result[denominator == 0] = 0
        return result

    sk.append(
        float(
            max(
                fb.areduce(aucs, reducer=fb.reducers.max),
                fb.areduce(aucs, reducer=fb.reducers.min),
            )
        )
    )
    a0.append(
        float(
            fb.areduce(aucs, reducer=fb.reducers.max)
            - fb.areduce(aucs, reducer=fb.reducers.min)
        )
    )
    a3.append(
        1
        - float(
            min(
                fb.areduce(aucs, reducer=fb.reducers.max),
                fb.areduce(aucs, reducer=fb.reducers.min),
            )
        )
    )
    a1.append(
        float(fb.areduce(aucs, reducer=fb.reducers.max, expand=fb.expanders.barea))
    )
    a2.append(
        float(
            fb.areduce(
                aucs,
                reducer=fb.reducers.max,
                expand=lambda values: fb.expanders.barea(values, comparator=biimpl),
            )
        )
    )

    if sk[-1] < 0.65:
        i1 = len(sk) - 1
        i1_aucs = aucs
    if sk[-1] < 0.98:
        i3 = len(sk) - 1
        i3_aucs = aucs


from matplotlib import pyplot as plt

"""
plt.subplot(1, 2, 1)
fb.visualize(i1_aucs.explain.curve, hold=True)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title(f'ABROCA {a1[i1]:.3f}     RBROCA {a2[i1]:.3f}')
plt.tight_layout()
plt.subplot(1, 2, 2)
fb.visualize(i3_aucs.explain.curve, hold=True, legend=False)
plt.title(f'ABROCA {a1[i3]:.3f}     RBROCA {a2[i3]:.3f}')
plt.xlabel('FPR')
#plt.ylabel('TPR')
plt.tight_layout()
plt.subplots_adjust(bottom=0.2, top=0.8)
plt.show()"""

# plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.plot(sk, a3, label="1-Worst AUC", color="blue")
plt.plot(sk, a0, label="AUC difference", color="green")
plt.plot(sk, a1, label="ABROCA", color="red")
plt.plot(sk, a2, label="RBROCA", color="purple")
plt.legend()
plt.xlabel("Best AUC")
plt.ylabel("Bias")

# Adding circle markers at the minimum points
"""
min_index_a3 = a3.index(min(a3))
plt.scatter(sk[min_index_a3], a3[min_index_a3], color='blue', marker='o')
min_index_a0 = a0.index(min(a0))
plt.scatter(sk[min_index_a0], a0[min_index_a0], color='green', marker='o')
min_index_a1 = a1.index(min(a1))
plt.scatter(sk[min_index_a1], a1[min_index_a1], color='red', marker='o')
min_index_a2 = a2.index(min(a2))
plt.scatter(sk[min_index_a2], a2[min_index_a2], color='purple', marker='o')"""


plt.show()
