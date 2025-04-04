from fairbench.v2.core import Value, Descriptor


class IndividualStamp:
    def __init__(self, name: str, sequence: str, details: str, caveats: list[str]):
        assert (
            details[-1] == "."
        ), f"Details did not end in a fullstop for stamp {name}."
        self.name = name
        self.sequence = sequence.split(".")
        self.details = details
        self.caveats = caveats


class Stamps:
    stamps = (
        IndividualStamp(
            "worst accuracy",
            "min.acc",
            details="This is the minimum benefit the system brings to any group.",
            caveats=[
                "The worst case is a lower bound but not an estimation of overall performance.",
                "There may be different distributions of benefits that could be protected.",
                "Ensure continuous monitoring and re-evaluation as group dynamics and external factors evolve.",
                "Ensure that high worst accuracy translates to meaningful benefits across all groups in the real-world context.",
                "Seek input from affected groups to understand the impact of errors and to inform remediation strategies.",
            ],
        ),
        IndividualStamp(
            "standard deviation",
            "stdx2.acc",
            details="This reflects imbalances in the distribution of correctness across groups, where correctness is measured with accuracy. The computed standard deviation is doubled, because this way the assessment value becomes 1 in the worst case, and remains 0 for perfect bias mitigation at equal accuracies.",
            caveats=[
                "Measuring only standard deviation may not always be an appropriate fairness consideration, and may obscure other important fairness concerns or create new disparities.",
                'Always consider trade-offs with overall or minimum accuracy, as the easiest way to "optimize" for this measure would be to degrade accuracy for all groups to the lowest level among groups.',
                "Ensure continuous monitoring and re-evaluation as group dynamics and external factors evolve.",
            ],
        ),
        IndividualStamp(
            "differential fairness",
            "maxrel.acc",
            details="The worst deviation of accuracy ratios from 1 is reported, so that value of 1 indicates disparate impact, and value of 0 disparate impact mitigation.",
            caveats=[
                "Disparate impact may not always be an appropriate fairness consideration, and may obscure other important fairness concerns or create new disparities.",
                'Always consider trade-offs with overall or minimum accuracy, as the easiest way to "optimize" for this measure would be to degrade accuracy for all groups to the lowest level among groups.',
                "Ensure continuous monitoring and re-evaluation as group dynamics and external factors evolve.",
            ],
        ),
        IndividualStamp(
            "max |Δfpr|",
            "maxdiff.tnr",
            details="The false positive rate differences are computed via the equivalent true negative rate differences. The maximum difference between pairs of groups is reported, so that value of 1 indicates disparate mistreatment, and value of 0 disparate mistreatment mitigation.",
            caveats=[
                "Disparate mistreatment may not always be an appropriate fairness consideration, and may obscure other important fairness concerns or create new disparities.",
                "Consider input from affected stakeholders to determine whether |Δfpr| is an appropriate fairness measure.",
                "Ensure continuous monitoring and re-evaluation as group dynamics and external factors evolve.",
                "Variations in FPR could be influenced by factors unrelated to the fairness of the system, such as data quality or representation.",
                "Mitigating |Δfpr| tends to mitigate |Δfnr|, and conversely.",
                "Seek input from affected groups to understand the impact of errors and to inform remediation strategies.",
            ],
        ),
        IndividualStamp(
            "max |Δfnr|",
            "maxdiff.tpr",
            details="The false negative rate differences are computed via the equivalent true positive rate differences. The maximum difference between pairs of groups is reported, so that value of 1 indicates disparate mistreatment, and value of 0 disparate mistreatment mitigation.",
            caveats=[
                "Disparate mistreatment may not always be an appropriate fairness consideration, and may obscure other important fairness concerns or create new disparities.",
                "Consider input from affected stakeholders to determine whether |Δfnr| is an appropriate fairness measure.",
                "Ensure continuous monitoring and re-evaluation as group dynamics and external factors evolve.",
                "Variations in FPR could be influenced by factors unrelated to the fairness of the system, such as data quality or representation.",
                "Mitigating |Δfpr| tends to mitigate |Δfnr|, and conversely.",
                "Seek input from affected groups to understand the impact of errors and to inform remediation strategies.",
            ],
        ),
        IndividualStamp(
            "max |Δfpr|",
            "largestmaxdiff.tnr",
            details="The false positive rate differences are computed via the equivalent true negative rate differences. The maximum difference between each group and the largest group (typically the whole population `all` is included in analysis that outputs this value) is reported, so that value of 1 indicates disparate mistreatment, and value of 0 disparate mistreatment mitigation.",
            caveats=[
                "Disparate mistreatment may not always be an appropriate fairness consideration, and may obscure other important fairness concerns or create new disparities.",
                "Consider input from affected stakeholders to determine whether |Δfpr| is an appropriate fairness measure.",
                "Ensure continuous monitoring and re-evaluation as group dynamics and external factors evolve.",
                "Variations in FPR could be influenced by factors unrelated to the fairness of the system, such as data quality or representation.",
                "Mitigating |Δfpr| tends to mitigate |Δfnr|, and conversely.",
                "Seek input from affected groups to understand the impact of errors and to inform remediation strategies.",
            ],
        ),
        IndividualStamp(
            "max |Δfnr|",
            "largestmaxdiff.tpr",
            details="The false negative rate differences are computed via the equivalent true positive rate differences. The maximum difference between each group and the largest group (typically the whole population `all` is included in analysis that outputs this value) is reported, so that value of 1 indicates disparate mistreatment, and value of 0 disparate mistreatment mitigation.",
            caveats=[
                "Disparate mistreatment may not always be an appropriate fairness consideration, and may obscure other important fairness concerns or create new disparities.",
                "Consider input from affected stakeholders to determine whether |Δfnr| is an appropriate fairness measure.",
                "Ensure continuous monitoring and re-evaluation as group dynamics and external factors evolve.",
                "Variations in FPR could be influenced by factors unrelated to the fairness of the system, such as data quality or representation.",
                "Mitigating |Δfpr| tends to mitigate |Δfnr|, and conversely.",
                "Seek input from affected groups to understand the impact of errors and to inform remediation strategies.",
            ],
        ),
        IndividualStamp(
            "max abroca",
            "maxbarea.auc",
            details="The maximum absolute between-ness area compares receiver operating characteristic (roc) curves pairwise and computes the area between them. Curves are retrieved from intermediate computations of the area under curve of the roc (auc). The absolute unsigned area is computed so that value of 0 indicates identical curves (otherwise the signed area would be the absolute auc difference).",
            caveats=[
                "Disparate mistreatment may not always be an appropriate fairness consideration, and may obscure other important fairness concerns or create new disparities.",
                "Consider input from affected stakeholders to determine whether abroca is an appropriate fairness measure.",
            ],
        ),
        IndividualStamp(
            "max abroca",
            "largestmaxbarea.auc",
            details="The maximum absolute between-ness area compares receiver operating characteristic (roc) curves pairwise and computes the area between each curve and the one of the largest group (typically the whole population `all` is included in analysis that outputs this value). Curves are retrieved from intermediate computations of the area under curve of the roc (auc). The absolute unsigned area is computed so that value of indicates identical curves (otherwise the signed area would be the absolute auc difference).",
            caveats=[
                "Disparate mistreatment may not always be an appropriate fairness consideration, and may obscure other important fairness concerns or create new disparities.",
                "Consider input from affected stakeholders to determine whether abroca is an appropriate fairness measure.",
            ],
        ),
    )

    def filter(self, value: Value) -> Value:
        results = list()
        for stamp in Stamps.stamps:
            try:
                val = value
                for element in stamp.sequence:
                    val = val[element]
                details = stamp.details
                val = val.rebase(
                    Descriptor(
                        stamp.name,
                        "stamp",
                        (
                            val.descriptor.details
                            + " of "
                            + value.descriptor.details
                            + "."
                            + "# Details\n"
                            + details
                            + (
                                "\n# Caveats and recommendations"
                                if stamp.caveats
                                else ""
                            )
                            + "".join("\n • " + caveat for caveat in stamp.caveats)
                            + "\n# Distribution"
                        ),
                    )
                )
                if val.exists():
                    results.append(val)
            except AssertionError:
                pass

        return Value(
            depends=results,
            descriptor=Descriptor(
                "fairness modelcard",
                "modelcard",
                "a modelcard that contains popular fairness stamps."
                "\nThese are obtained from "
                + value.descriptor.details
                + "\nStamps contain caveats and recommendation that should be considered during practical adoption. "
                "They are only a part of the full analysis that has been conducted, so consider also viewing "
                "the full generated report to find more prospective biases.\n",
            ),
        )
