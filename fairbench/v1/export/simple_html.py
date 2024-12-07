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
        return {k + _desc(v): fork_to_dict(v) for k, v in fork.branches().items()}
    if isinstance(fork, DotDict):
        return {k + _desc(v): fork_to_dict(v) for k, v in fork.items()}
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
        return None
    return float(tobackend(fork).numpy())


def simple_html(fork, name="Report", filename=None, show=True):
    # Convert the fork structure to a JSON string
    json_data = json.dumps(fork_to_dict(fork), indent=4)

    # Step 2: Generate HTML Content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{name}</title>
        <style>
            #container {{
                width: 800px;
                margin: 20px auto;
                position: relative;
            }}
            .branch {{
                margin-left: 10px;
                cursor: pointer;
                position: relative;
                display: block;
            }}
            .label {{
                min-width: 120px;
                display: inline-block;
                vertical-align: middle;
                margin-right: 10px;
            }}
            .expand-icon {{
                display: inline-block;
                width: 20px;
                height: 20px;
                background-color: lightgray;
                color: black;
                text-align: center;
                vertical-align: middle;
                margin-right: 5px;
                border-radius: 50%;
                font-weight: bold;
            }}
            .hidden {{
                display: none;
            }}
            .bar {{
                height: 10px; /* Thinner bars */
                display: inline-block;
                vertical-align: middle;
                margin-left: 10px;
                width: 0; /* Bar width will be set dynamically */
            }}
            .value {{
                display: inline-block;
                vertical-align: middle;
                margin-right: 10px;
                min-width: 60px; /* Reserve space for the value */
                text-align: right;
            }}
            .children {{
                margin-left: 20px; /* Indent child elements */
                margin-top: 10px;
                margin-bottom: 10px;
            }}
            .bar-container {{
                display: inline-block;
                min-width: 300px; /* Ensure all bars start from the same position */
                margin-left: 0px; /* Ensure bars start from the same horizontal position */
                vertical-align: middle;
            }}
            #collapseButton {{
                width: 120px;
                margin: 20px auto;
                padding: 10px;
                background-color: #D22B2B;
                color: white;
                text-align: center;
                cursor: pointer;
                border: none;
                border-radius: 5px;
                font-size: 16px;
            }}
        </style>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js" integrity="sha384-MrcW6ZMFYlzcLA8Nl+NtUVF0sA7MsXsP1UyJoMp4YLEuNSfAP+JcXn/tWtIaxVXM" crossorigin="anonymous"></script>
        <script src="https://d3js.org/d3.v7.min.js"></script>
    </head>
    <body>
        <div id="container">
            <h1>{name}</h1>
            <p>Click on an entry to expand collapse explanations.</p>
            <button id="collapseButton">Collapse all</button>
            <div id="visualization"></div>
        </div>

        <script>
            const data = {json_data};

            // Function to get color based on the level
            function getColor(level) {{
                const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']; // Define colors for up to 5 levels
                return colors[level % colors.length];
            }}

            function toggleLabel(element) {{
                // Toggle the associated children of the clicked label
                const parentDiv = element.parentElement;
                const childrenDiv = parentDiv.querySelector('.children');
                if (childrenDiv) {{
                    childrenDiv.classList.toggle('hidden');
                    const icon = parentDiv.querySelector('.expand-icon');
                    if (icon) {{
                        icon.innerHTML = childrenDiv.classList.contains('hidden') ? '&#8250;' : '-';
                    }}
                }}
            }}

            function createVisualization(data, container, expand=true, level=0) {{
                if (typeof data === 'number') {{
                    // If the data is a single number, treat it as a root-level value
                    const div = document.createElement('div');
                    div.className = 'branch';

                    const valueToDisplay = data > 1 ? Math.round(data) : data.toFixed(3);

                    // Create value display
                    const valueDiv = document.createElement('div');
                    valueDiv.className = 'value';
                    valueDiv.textContent = valueToDisplay;
                    div.appendChild(valueDiv);

                    if (data <= 1) {{
                        // Create bar for values <= 1
                        const barDiv = document.createElement('div');
                        barDiv.className = 'bar';
                        barDiv.style.width = (data * 100) + 'px';
                        barDiv.style.backgroundColor = getColor(level);

                        const barContainer = document.createElement('div');
                        barContainer.className = 'bar-container';
                        barContainer.appendChild(barDiv);
                        div.appendChild(barContainer);
                    }}

                    container.appendChild(div);
                    return;
                }}

                for (const [key, value] of Object.entries(data)) {{
                    if (value === null) continue; // Skip null values

                    const div = document.createElement('div');
                    div.className = 'branch';

                    const labelDiv = document.createElement('div');
                    labelDiv.className = 'label';
                    labelDiv.textContent = key;
                    div.appendChild(labelDiv);

                    // Always display "samples" and "top" as integers without bars
                    if (typeof value === 'number') {{
                        const valueDiv = document.createElement('div');
                        valueDiv.className = 'value';
                        valueDiv.textContent = Math.round(value);
                        div.appendChild(valueDiv);
                    }} else if (value && typeof value === 'object' && value.hasOwnProperty('value')) {{
                        const valueToDisplay = value.value > 1 ? Math.round(value.value) : value.value.toFixed(3);

                        // Create value display
                        const valueDiv = document.createElement('div');
                        valueDiv.className = 'value';
                        valueDiv.textContent = valueToDisplay;
                        div.appendChild(valueDiv);

                        if (value.value <= 1) {{
                            // Create bar for values <= 1
                            const barDiv = document.createElement('div');
                            barDiv.className = 'bar';
                            barDiv.style.width = (value.value * 100) + 'px';
                            barDiv.style.backgroundColor = getColor(level);

                            const barContainer = document.createElement('div');
                            barContainer.className = 'bar-container';
                            barContainer.appendChild(barDiv);
                            div.appendChild(barContainer);
                        }}

                        // Create the expandable "explain" section
                        if (value.hasOwnProperty('explain')) {{
                            const expandIcon = document.createElement('div');
                            expandIcon.className = 'expand-icon';
                            expandIcon.innerHTML = '&#8250;';
                            div.insertBefore(expandIcon, labelDiv);

                            const explainDiv = document.createElement('div');
                            explainDiv.className = 'children hidden';
                            createVisualization(value.explain, explainDiv, false, level + 1);
                            div.appendChild(explainDiv);

                            expandIcon.onclick = labelDiv.onclick = function (e) {{
                                e.stopPropagation(); // Prevent clicking from affecting parent elements
                                toggleLabel(labelDiv);
                            }};
                        }}
                    }} else if (typeof value === 'object' && value !== null) {{
                        // Handle other objects recursively
                        const expandIcon = document.createElement('div');
                        expandIcon.className = 'expand-icon';
                        expandIcon.innerHTML = '&#8250;';
                        div.insertBefore(expandIcon, labelDiv);

                        const childDiv = document.createElement('div');
                        childDiv.className = 'children hidden';
                        createVisualization(value, childDiv, expand, level + 1);
                        div.appendChild(childDiv);

                        expandIcon.onclick = labelDiv.onclick = function (e) {{
                            e.stopPropagation(); // Prevent clicking from affecting parent elements
                            toggleLabel(labelDiv);
                        }};
                    }}

                    container.appendChild(div);

                    // Expand all levels by default until bars are shown
                    if (expand && !div.querySelector('.bar')) {{
                        const childrenDiv = div.querySelector('.children');
                        if (childrenDiv) {{
                            childrenDiv.classList.remove('hidden');
                            const icon = div.querySelector('.expand-icon');
                            if (icon) {{
                                icon.innerHTML = '-';
                            }}
                        }}
                    }}
                }}
            }}

            function collapseAll() {{
                const childrenDivs = document.querySelectorAll('.children');
                childrenDivs.forEach(childrenDiv => {{
                    childrenDiv.classList.add('hidden');
                    const parentDiv = childrenDiv.parentElement;
                    const icon = parentDiv.querySelector('.expand-icon');
                    if (icon) {{
                        icon.innerHTML = '&#8250;';
                    }}
                }});
            }}

            const container = document.getElementById('visualization');
            createVisualization(data, container);

            document.getElementById('collapseButton').onclick = collapseAll;
        </script>
    </body>
    </html>
    """

    # Step 3: Save the HTML Content to a File or Create a Temporary File and Open in Browser
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
