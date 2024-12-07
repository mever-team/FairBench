from fairbench.v1.export.modelcards.toyaml import toyamlprimitives
from fairbench.v1.core import Fork, Explainable, ExplainableError


def _resulttomarkdown(value, inline=False):
    if value == "False":
        return ":x:"
    if value == "True":
        return ":heavy_check_mark:"
    if inline:
        return f"`{value}`"
    return value


def tomarkdown(report, _=None, table=True, file=None):
    assert _ is None, "Explicitly use keyword arguments."
    prepend = ""
    ret = "## Metrics\n"
    if table:
        ret += "- Fairness-aware metrics are computed. "
    factors = list()
    caveats = list()
    for row in toyamlprimitives(report):
        results = _resulttomarkdown(row["results"], inline=not table)
        metric_fork = (
            report[row["name"]]
            if not isinstance(report, Fork)
            else getattr(report, row["name"])
        )
        while isinstance(metric_fork, Explainable):
            metric_fork = metric_fork.explain
        if isinstance(metric_fork, Fork):
            metric_fork = metric_fork.branches()
        if isinstance(metric_fork, ExplainableError):
            raise Exception(metric_fork.explain)
            # continue
        factors.extend(metric_fork.keys())
        if table:
            caveats.extend(row["caveats"])
            if not prepend:
                prepend += "\n\n## Evaluation Results\n"
                prepend += "| Metric | Value |\n"
                prepend += "| ------ | ----- |\n"
            prepend += f'| {row["name"]} | {results} |\n'
            ret += f"The *{row['name']}* {row['description'][0].lower()}{row['description'][1:]} "
        else:
            ret += f"### {row['name']} {results}\n{row['description']} {' '.join(row['caveats'])}\n\n"
    if factors:
        factors = sorted(set(factors))
        factor_text = "## Factors\n"
        factor_text += (
            "- The groups that are considered for fairness assessment are "
            + (", ".join(factors))
            + "."
        )
        factor_text += "\n\n"
    else:
        factor_text = ""
    if caveats:
        caveats = sorted(set(caveats))
        caveats_text = "\n## Caveats and Recommendations\n- "
        caveats_text += "\n- ".join(caveats)
        caveats_text += "\n\n"
    else:
        caveats_text = ""
    ret = factor_text + ret + prepend + caveats_text
    if file is not None:
        with open(file, "w") as file:
            file.write(ret)
    return ret
