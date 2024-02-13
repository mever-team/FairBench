import scipy
import numpy as np
import matplotlib.pyplot as plt
import tqdm

x = list()
y1 = list()
y2 = list()
for i in tqdm.tqdm(range(300)):
    samples = 1000
    minority = np.random.uniform(0.1, 0.5)
    positive_rate = np.random.uniform(0.1, 0.5)
    bias = i / 300.0 * 0.1
    num_samples = 1

    plt.subplot(2, 1, 1)
    sensitive = np.random.choice([0, 1], size=samples, p=[1 - minority, minority])
    positives = np.array(
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

    def distance(sens, nonsens):
        sens_counts = [np.sum(sens == 1), np.sum(sens == 0)]
        nonsens_counts = [np.sum(nonsens == 1), np.sum(nonsens == 0)]
        contingency = [sens_counts, nonsens_counts]
        return 1 - scipy.stats.chi2_contingency(contingency)[1]

    sens = positives[sensitive == 1]
    nonsens = positives[sensitive == 0]

    encounter_minority = np.mean(
        sensitive
    )  # distance(positives[sensitive==1], (1-positives)[sensitive==0])
    encounter_minority = 1 - (1 - encounter_minority) ** num_samples

    # discrimination = 1-scipy.stats.ttest_ind(positives[sensitive==1], positives[sensitive==0], equal_var=False).pvalue
    discrimination = distance(sens, nonsens)
    found_bias = max(0, encounter_minority + discrimination - 1)
    prule = min(
        np.mean(positives[sensitive == 1]) / np.mean(positives[sensitive == 0]),
        np.mean(positives[sensitive == 0]) / np.mean(positives[sensitive == 1]),
    )
    # print(f'Minority  {encounter_minority:.3f}')
    # print(f'Imbalance {discrimination:.3f}')
    # print(f'prule     {min(np.mean(positives[sensitive==1])/np.mean(positives[sensitive==0]), np.mean(positives[sensitive==0])/np.mean(positives[sensitive==1])):.3f}')
    # print(f'Bias      {max(0, encounter_minority+discrimination-1):.3f}')
    x.append(bias)
    y1.append(found_bias)
    y2.append(max(0, 0.8 - prule))

window = 30
plt.scatter(
    [
        np.mean([x[j] for j in range(i, i + window)])
        for i in range(0, len(x) - window, window)
    ],
    [
        len([y1[j] for j in range(i, i + window) if y1[j] > 0]) / window
        for i in range(0, len(y1) - window, window)
    ],
    label="found bias",
)
plt.scatter(
    [
        np.mean([x[j] for j in range(i, i + window)])
        for i in range(0, len(x) - window, window)
    ],
    [
        len([y2[j] for j in range(i, i + window) if y2[j] > 0]) / window
        for i in range(0, len(y2) - window, window)
    ],
    label="1-prule",
)
plt.legend()
# plt.scatter(x, np.array(y1)/np.array(y2))
plt.show()
