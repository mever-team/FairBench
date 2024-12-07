from fairbench.v1.export.modelcards.toyaml import toyamlprimitives
import webbrowser
import tempfile
from fairbench.v1.core import Fork, Explainable, ExplainableError


def _resulttohtml(value, inline=False):
    if value == "False":
        return "&#10060;"
    if value == "True":
        return "&#9989;"
    if inline:
        return value
    return value


def tohtml(report, _=None, table=True, file=None, show=False):
    assert _ is None, "Explicitly use keyword arguments."
    prepend = ""
    factors = list()
    caveats = list()
    ret = "<h2>Metrics</h2>"
    if table:
        ret += "<ul><li>Fairness-aware metrics are computed. "
    for row in toyamlprimitives(report):
        results = _resulttohtml(row["results"], inline=not table)
        desc = row["description"]
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
        factors.extend(metric_fork.keys())
        if table:
            caveats.extend(row["caveats"])
            if not prepend:
                prepend += "\n<h2>Evaluation Results</h2>\n<table>\n"
                prepend += "<tr><th>Metric</th><th>Value</th></tr>\n"
            prepend += f'<tr><td>{row["name"]}</td><td>{results}</td></tr>\n'
            ret += f"The <em>{row['name']}</em> {desc[0].lower()}{desc[1:]} "
        else:
            ret += f"<h3>{row['name']} {results}</h3>\n{desc} {' '.join(row['caveats'])}\n\n"
    if table:
        ret += "</li></ul>"
        prepend += "</table>"
    if factors:
        factors = sorted(set(factors))
        factor_text = "<h2>Factors</h2>"
        factor_text += (
            "<ul><li>The groups that are considered for fairness assessment are "
            + (", ".join(factors))
            + "."
        )
        factor_text += "</li></ul>"
    else:
        factor_text = ""
    if caveats:
        caveats = sorted(set(caveats))
        caveats_text = "<h2>Caveats and Recommendations</h2>"
        caveats_text += "<ul>"
        for caveat in caveats:
            caveats_text += "<li>" + caveat + "</li>"
        caveats_text += "</ul>"
    else:
        caveats_text = ""
    ret = factor_text + ret + prepend + caveats_text
    if file is not None:
        with open(file, "w") as wfile:
            wfile.write(ret)
        if show:
            webbrowser.open_new(file)
    elif show:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html") as wfile:
            url = "file://" + wfile.name
            wfile.write(ret)
        webbrowser.open(url)
    return ret
