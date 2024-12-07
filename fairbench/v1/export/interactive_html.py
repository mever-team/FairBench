import json
import tempfile
import webbrowser
from fairbench.v1.core import (
    Fork,
    DotDict,
    Explainable,
    ExplainableError,
    tobackend,
    ExplanationCurve,
)


def _desc(v):
    ret = None
    if isinstance(v, DotDict):
        ret = v.role()
    if isinstance(v, Fork):
        ret = v.role()
    if ret is None:
        return ""
    return ret


def fork_to_dict(fork):
    if isinstance(fork, Fork):
        return {
            k + _desc(v): fork_to_dict(v)
            for k, v in fork.branches().items()
            if v is not None
        }
    if isinstance(fork, DotDict):
        return {k + _desc(v): fork_to_dict(v) for k, v in fork.items() if v is not None}
    if isinstance(fork, ExplainableError):
        return None
    if isinstance(fork, Explainable):
        return {
            "value": fork_to_dict(fork.value),
            "explain": fork_to_dict(fork.explain),
        }
    if isinstance(fork, float) or isinstance(fork, int):
        return fork
    if isinstance(fork, ExplanationCurve):
        return {
            "curve": {
                "x": fork.x.tolist(),
                "y": fork.y.tolist(),
                "name": fork.name,
            }
        }
    return float(tobackend(fork).numpy())


def interactive_html(fork, name="Report", filename=None, show=True):
    import json
    import tempfile
    import webbrowser

    # Convert the fork structure to a JSON string
    json_data = json.dumps(fork_to_dict(fork), indent=4)

    tooltip = {
        "accuracy": "The fraction of correct predictions",
        "pr": "The fraction of positive predictions",
        "tpr": "The false positive rates",
        "tnr": "The true negative rates",
        "fpr": "The false positive rates",
        "fnr": "The false negative rates",
        "min": "The minimum value for metrics across all groups",
        "minratio": "THe minimum when dividing the metric values of groups",
        "maxdiff": "The maximum difference between groups",
        "wmean": "The mean metric value weighted by group size",
        "max": "The maximum value across all comparisons",
        "gini": "The gini score of metric valeus across groups",
        "maxbarea": "The maximum betweeness area of group curves",
        "maxbdcg": "The maximum ndcg-weighted difference of group curves",
        "maxrarea": "The maximum integral of 1-value ratio of group curves",
        "avgscore": "The average of scores",
        "avgrepr": "The average representations in top-k recommendations",
        "tophr": "The hit rate among the top-k recommendations",
        "toprec": "The number in the top-k recommend items",
        "auc": "THe area under curve of the receiver operating characteristics",
        "samples": "The number of groups samples",
        "positive": "The number of positive samples",
        "negatives": "The number of negative sample",
        "true": "The number of true labels",
        "true_positives": "The number of positives that are actually so",
        "true_negatives": "The number of negatives that are actually so",
        "false": "The number of false labels",
        "curve": "The curve used to compute values",
    }
    tooltip_vsany = {k + "[vsAny]": v for k, v in tooltip.items()}
    tooltip = {**tooltip, **tooltip_vsany}
    tooltip_json = json.dumps(tooltip, indent=4)

    # Generate HTML Content
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{name}</title>
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script>
        document.addEventListener(\\DOMContentLoaded", function() {{
            var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
            tooltipTriggerList.forEach(function(tooltipTriggerEl) {{
                new bootstrap.Tooltip(tooltipTriggerEl);
            }});
        }});
    </script>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container mt-5">
        <h3>
            <nav aria-label="breadcrumb">
                <ol id="breadcrumb" class="breadcrumb"></ol>
            </nav>
        </h3>
        The main report contains a summary that gets
        progressively into more details as you focus on specific (sub)entries by selecting them.
        Hover over the entries to get a proper description of what the acronyms mean.
        Use <i>expand all</i> to compare the contents of all currently visible info, and click on
        the path above to go back to a previous step.<br><br>
        <div id="chart"></div>
        <button id="compareButton" class="btn btn-info mt-3" style="display: none;">Expand all</button>
    </div>

    <script>
        const data = {json_data};
        const tooltip = {tooltip_json}; 

        let historyStack = [];
        let currentData = data;
        let pathStack = [{{
            name: '{name}',
            data: data
        }}];

        function getNumericValue(value) {{
            if (typeof value === 'number') {{
                return value;
            }} else if (value !== null && typeof value === 'object') {{
                if ('value' in value && typeof value.value === 'number') {{
                    return value.value;
                }}
            }}
            return null;
        }}

        function renderBreadcrumb() {{
            const breadcrumbOl = document.getElementById('breadcrumb');
            breadcrumbOl.innerHTML = '';

            pathStack.forEach((item, index) => {{
                const li = document.createElement('li');
                li.className = 'breadcrumb-item';

                if (index === pathStack.length - 1) {{
                    li.classList.add('active');
                    li.setAttribute('aria-current', 'page');
                    li.textContent = item.name;
                }} else {{
                    const a = document.createElement('a');
                    a.href = '#';
                    a.textContent = item.name;
                    a.onclick = function(e) {{
                        e.preventDefault();
                        // Navigate back to this point in the history
                        const stepsBack = pathStack.length - index - 1;
                        for (let i = 0; i < stepsBack; i++) {{
                            historyStack.pop();
                            pathStack.pop();
                        }}
                        currentData = item.data;
                        render(currentData);
                    }};
                    li.appendChild(a);
                }}
                breadcrumbOl.appendChild(li);
            }});
        }}

        function render(data) {{
            const chartDiv = document.getElementById('chart');
            chartDiv.innerHTML = ''; // Clear previous content

            renderBreadcrumb();

            // Hide or show the "Expand all" button
            const compareButton = document.getElementById('compareButton');
            compareButton.style.display = 'none';

            // If data is a number, display it
            if (typeof data === 'number') {{
                const p = document.createElement('p');
                p.textContent = data;
                chartDiv.appendChild(p);
                return;
            }}

            // Check if data is an empty object
            if (data && typeof data === 'object' && Object.keys(data).length === 0) {{
                const p = document.createElement('p');
                p.textContent = 'No data available';
                chartDiv.appendChild(p);
                return;
            }}

            const entries = Object.entries(data);

            // Create a container div
            const container = document.createElement('div');
            container.className = 'row';

            let hasExplanations = false;

            entries.forEach(([key, value]) => {{
                if (value === null || value === undefined) {{
                    return; // Skip entries with None values
                }}

                // Create a div for each item
                const itemDiv = document.createElement('div');
                itemDiv.className = 'col-12 mb-2 d-flex align-items-center';

                // Create label button
                const labelButton = document.createElement('button');
                labelButton.className = 'btn btn-light me-2 text-start';
                labelButton.style.width = '200px';
                labelButton.style.textDecoration = 'none';
                labelButton.style.padding = '0';
                labelButton.textContent = key;
                
                // Set tooltip based on key
                labelButton.setAttribute('data-bs-toggle', 'tooltip');
                labelButton.setAttribute('data-bs-placement', 'top');  // Position tooltip on top
                labelButton.title = tooltip[key] || "Sensitive dimension";

                // Check if the value can be navigated further
                let isNavigable = false;

                if (value !== null && typeof value === 'object') {{
                    if (!('curve' in value)) {{
                        isNavigable = true;
                    }}
                }}

                if (isNavigable) {{
                    labelButton.onclick = function() {{
                        if (value !== null && typeof value === 'object') {{
                            historyStack.push(currentData);
                            pathStack.push({{ name: key, data: value }});
                            if (value.explain) {{
                                currentData = value.explain;
                            }} else {{
                                currentData = value;
                            }}
                            render(currentData);
                        }}
                    }};
                    hasExplanations = true;
                }}

                itemDiv.appendChild(labelButton);

                // Handle curve data
                if (value !== null && typeof value === 'object' && 'curve' in value) {{
                    const canvas = document.createElement('canvas');
                    canvas.style.maxWidth = '400px';
                    canvas.style.height = '200px'; // Make the chart taller to be almost square
                    labelButton.style.height = '200px';
                    itemDiv.appendChild(canvas);

                    const ctx = canvas.getContext('2d');
                    new Chart(ctx, {{
                        type: 'line',
                        data: {{
                            labels: value.curve.x,
                            datasets: [{{
                                label: value.curve.name,
                                data: value.curve.y,
                                borderColor: 'red',
                                borderWidth: 1,
                                fill: false,
                            }}],
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {{
                                x: {{
                                    display: true,
                                }},
                                y: {{
                                    display: true,
                                }},
                            }},
                            plugins: {{
                                legend: {{
                                    display: false,
                                }},
                                tooltip: {{
                                    enabled: true,
                                }},
                            }},
                            elements: {{
                                line: {{
                                    tension: 0, // Disable bezier curves
                                }},
                            }},
                        }},
                    }});
                }} else {{
                    // Get numeric value
                    const numericValue = getNumericValue(value);

                    if (numericValue !== null) {{
                        // Create bar
                        const bar = document.createElement('div');
                        bar.className = 'progress flex-grow-1';

                        const progressBar = document.createElement('div');
                        progressBar.className = 'progress-bar bg-secondary rounded-0'; // Make bar rectangular and gray
                        progressBar.style.textAlign = 'left';

                        const widthPercent = Math.min(Math.max(numericValue * 100, 0), 100); // Clamp between 0 and 100
                        progressBar.style.width = widthPercent + '%';
                        progressBar.textContent = numericValue.toFixed(2);

                        bar.appendChild(progressBar);

                        if (isNavigable) {{
                            bar.style.cursor = 'pointer';
                            bar.onclick = function() {{
                                if (value !== null && typeof value === 'object') {{
                                    historyStack.push(currentData);
                                    pathStack.push({{ name: key, data: value }});
                                    if (value.explain) {{
                                        currentData = value.explain;
                                    }} else {{
                                        currentData = value;
                                    }}
                                    render(currentData);
                                }}
                            }};
                        }}

                        itemDiv.appendChild(bar);
                    }}
                }}

                container.appendChild(itemDiv);
            }});

            chartDiv.appendChild(container);

            // Show "Expand all" button if there are explanations to expand
            if (hasExplanations) {{
                compareButton.style.display = 'block';
                compareButton.onclick = function() {{
                    // Hide the compare button
                    compareButton.style.display = 'none';

                    // Save the current state to history
                    historyStack.push(currentData);
                    pathStack.push({{ name: 'expanded', data: currentData }});

                    // Render the expanded view
                    renderComparison(entries.filter(([key, value]) => {{
                        // Only include navigable items
                        if (value === null || value === undefined) {{
                            return false;
                        }}
                        if (typeof value === 'object' && !('curve' in value)) {{
                            return true;
                        }}
                        return false;
                    }}));
                }};
            }}
        }}

        function renderComparison(items) {{
            const chartDiv = document.getElementById('chart');
            chartDiv.innerHTML = ''; // Clear previous content

            items.forEach(([key, value]) => {{
                const title = document.createElement('h3');
                title.textContent = key;
                chartDiv.appendChild(title);

                const container = document.createElement('div');
                container.className = 'row';

                // Render the explanation
                if (value.explain) {{
                    renderSubExplanation(value.explain, container, [key]);
                }} else {{
                    renderSubExplanation(value, container, [key]);
                }}

                chartDiv.appendChild(container);
            }});

            renderBreadcrumb();
        }}

        function renderSubExplanation(data, container, parentPath) {{
            if (typeof data !== 'object' || data === null) {{
                const p = document.createElement('p');
                p.textContent = data;
                container.appendChild(p);
                return;
            }}

            const entries = Object.entries(data);

            entries.forEach(([key, value]) => {{
                if (value === null || value === undefined) {{
                    return; // Skip entries with None values
                }}

                const itemDiv = document.createElement('div');
                itemDiv.className = 'col-12 mb-2 d-flex align-items-center';
                
                // Create label button
                const labelButton = document.createElement('button');
                labelButton.className = 'btn btn-light me-2 text-start';
                labelButton.style.width = '200px';
                labelButton.style.textDecoration = 'none';
                labelButton.style.padding = '0';
                labelButton.textContent = key;
                
                // Set tooltip based on key
                labelButton.setAttribute('data-bs-toggle', 'tooltip');
                labelButton.setAttribute('data-bs-placement', 'top');  // Position tooltip on top
                labelButton.title = tooltip[key] || "Sensitive dimension";

                // Check if the value can be navigated further
                let isNavigable = false;

                if (value !== null && typeof value === 'object') {{
                    if (!('curve' in value)) {{
                        isNavigable = true;
                    }}
                }}

                if (isNavigable) {{
                    labelButton.onclick = function() {{
                        // Save current data and path
                        historyStack.push(currentData);
                        pathStack.push({{ name: key, data: value }});

                        currentData = value.explain ? value.explain : value;
                        render(currentData);
                    }};
                }}

                itemDiv.appendChild(labelButton);

                // Handle curve data
                if (value !== null && typeof value === 'object' && 'curve' in value) {{
                    const canvas = document.createElement('canvas');
                    canvas.style.maxWidth = '400px';
                    canvas.style.height = '200px'; // Make the chart taller to be almost square
                    labelButton.style.height = '200px';
                    itemDiv.appendChild(canvas);

                    const ctx = canvas.getContext('2d');
                    new Chart(ctx, {{
                        type: 'line',
                        data: {{
                            labels: value.curve.x,
                            datasets: [{{
                                label: value.curve.name,
                                data: value.curve.y,
                                borderColor: 'red',
                                borderWidth: 1,
                                fill: false,
                            }}],
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {{
                                x: {{
                                    display: true,
                                }},
                                y: {{
                                    display: true,
                                }},
                            }},
                            plugins: {{
                                legend: {{
                                    display: false,
                                }},
                                tooltip: {{
                                    enabled: true,
                                }},
                            }},
                            elements: {{
                                line: {{
                                    tension: 0, // Disable bezier curves
                                }},
                            }},
                        }},
                    }});
                }} else {{
                    // Get numeric value
                    const numericValue = getNumericValue(value);

                    if (numericValue !== null) {{
                        // Create bar
                        const bar = document.createElement('div');
                        bar.className = 'progress flex-grow-1';

                        const progressBar = document.createElement('div');
                        progressBar.className = 'progress-bar bg-secondary rounded-0'; // Make bar rectangular and gray
                        progressBar.style.textAlign = 'left';

                        const widthPercent = Math.min(Math.max(numericValue * 100, 0), 100); // Clamp between 0 and 100
                        progressBar.style.width = widthPercent + '%';
                        progressBar.textContent = numericValue.toFixed(2);

                        bar.appendChild(progressBar);

                        if (isNavigable) {{
                            bar.style.cursor = 'pointer';
                            bar.onclick = function() {{
                                // Save current data and path
                                historyStack.push(currentData);
                                pathStack.push({{ name: key, data: value }});

                                currentData = value.explain ? value.explain : value;
                                render(currentData);
                            }};
                        }}

                        itemDiv.appendChild(bar);
                    }}
                }}

                container.appendChild(itemDiv);
            }});
        }}

        render(currentData);
    </script>
    <br>
    <br>
</body>
</html>
    """

    # Save the HTML Content to a File or Create a Temporary File and Open in Browser
    if filename is not None:
        with open(filename, "w") as file:
            file.write(html_content)
        if show:
            webbrowser.open_new_tab(filename)
    elif show:
        # with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html") as tmp_file:
        with open("temp.html", "w") as tmp_file:
            tmp_file.write(html_content)
            temp_filename = tmp_file.name
        webbrowser.open_new_tab(temp_filename)

    return html_content
