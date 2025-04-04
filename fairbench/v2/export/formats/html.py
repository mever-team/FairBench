import webbrowser
import json


class Html:
    chart_count = 0

    def __init__(
        self,
        horizontal=False,
        view=True,
        filename="temp",
        distributions=False,
        horizontal_bars=True,
        legend=True,
    ):
        self.contents = ""
        self.bars = []
        self.prev_max_level = 0
        self.routes = dict()
        self.curves = list()
        self.horizontal = horizontal
        self.level = 0
        self.view = view
        self.filename = filename
        self.distributions = distributions
        self.horizontal_bars = horizontal_bars
        self.legend = legend

    def navigation(self, text, routes: dict):
        return self

    def list(self, title, keys):
        self.contents += "\n" + title + "<br>"
        keys = sorted([str(key) for key in keys])
        for key in keys:
            self.contents += "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" + key + "<br>"
        return self

    def title(self, text, level=0, link=None):
        if self.bars:
            self._embed_bars()
        if self.curves:
            self._embed_curves()
        level = level * 2 + 1
        self.level = level
        if level > 6:
            level = 6
        if text:
            if level == 3:
                if self.prev_max_level >= level:
                    self.contents += "</div></div>"
                else:
                    self.contents += "<br>"
                self.contents += f'<div style="width: {400 if self.horizontal else 800}px; float: left;" class="card m-3">'
                # if self.legend:
                self.contents += f'<h{level} class="mt-0 text-white bg-dark p-3 rounded">{text}</h{level}>'

                self.contents += """
                <div class="card-body">
                """
            elif level <= 1:
                self.contents += f'<h{level} class="text-dark">{text}</h{level}>'
            else:
                if level == 5:
                    self.contents += "<hr>"
                if self.legend or level <= 2:
                    self.contents += (
                        f'<h{level} class="mt-3 text-dark"><b>{text}</b></h{level}>'
                    )
        self.prev_max_level = max(self.prev_max_level, level)
        return self

    def curve(self, title, x, y, units):
        if units == title:
            units = ""
        self.curves.append((title, x, y, units))
        return self

    def bar(self, title, val, target, units=""):
        if units == title:
            units = ""
        if units:
            units = "\n(" + units + ")"
        self.bars.append((title + units, val, target))
        return self

    def _embed_curves(self):
        if not self.distributions:
            self.contents += (
                f"<details><summary>Obtained from {len(self.bars)} curves</summary>\n"
            )
        curve_data = [
            {
                "title": t,
                "x": [float(val) for val in x],
                "y": [float(val) for val in y],
                "units": u,
            }
            for t, x, y, u in self.curves
        ]
        curve_json = json.dumps(curve_data)
        Html.chart_count += 1
        cid = Html.chart_count

        self.contents += f"""<div id="curve-chart{cid}" class="mt-2"></div>"""
        self.contents += f"""
            <script>
                const curveData{cid} = {curve_json};
                const margin{cid} = {{top: 20, right: 30, bottom: 50, left: 50 }};
                const width{cid} = {400 if self.horizontal else 600} - margin{cid}.left - margin{cid}.right;
                const height{cid} = 200 - margin{cid}.top - margin{cid}.bottom;
            
                const svg{cid} = d3.select("#curve-chart{cid}")
                      .append("svg")
                      .attr("width", width{cid} + margin{cid}.left + margin{cid}.right)
                      .attr("height", height{cid} + margin{cid}.top + margin{cid}.bottom)
                      .append("g");
            
                const xScale{cid} = d3.scaleLinear()
                    .domain([0, d3.max(curveData{cid}, d => d3.max(d.x))])
                    .range([0, width{cid}]);
            
                const yScale{cid} = d3.scaleLinear()
                    .domain([0, d3.max(curveData{cid}, d => d3.max(d.y))])
                    .nice()
                    .range([height{cid}, 0]);
            
                svg{cid}.append("g")
                    .call(d3.axisBottom(xScale{cid}));
            
                svg{cid}.append("g")
                    .call(d3.axisLeft(yScale{cid}));
            
                curveData{cid}.forEach((curve) => {{
                    const line{cid} = d3.line()
                        .x((_, i) => xScale{cid}(curve.x[i]))
                        .y((_, i) => yScale{cid}(curve.y[i]));
            
                    svg{cid}.append("path")
                        .datum(curve)
                        .attr("fill", "none")
                        .attr("stroke", "steelblue")
                        .attr("stroke-width", 1.5)
                        .attr("d", line{cid});
                }});
            
                svg{cid}.selectAll(".curve-label")
                    .data(curveData{cid})
                    .enter()
                    .append("text")
                    .attr("class", "curve-label")
                    .attr("x", d => xScale{cid}(d.x[Math.floor(d.x.length / 2)]) - 5)
                    .attr("y", d => yScale{cid}(d.y[Math.floor(d.y.length / 2)]) - 5)
                    .text(d => d.title + " (" + d.units + ")")
                    .style("font-size", "10px")
                    .style("fill", "black");
            </script>
        """

        if not self.distributions:
            self.contents += "\n</details>\n"
        self.curves = list()
        return self

    def _embed_bars(self):
        if not self.distributions:
            self.contents += (
                f"<details><summary>Obtained from {len(self.bars)} values</summary>\n"
            )
        bar_data = [
            {
                "title": (f"{v:.3f} " if v < 1 else str(int(v))) + t,
                "val": v,
                "target": trg,
            }
            for t, v, trg in self.bars
        ]
        max_val = max(v for t, v, trg in self.bars)
        if max_val < 1:
            max_val = 1
        bar_json = str(bar_data).replace("'", '"')
        Html.chart_count += 1
        cid = Html.chart_count
        self.contents += f"""<div id="bar-chart{cid}" class="mt-2"></div>"""
        if self.horizontal_bars:
            self.contents += f"""
                <script>
                    const data{cid} = {bar_json};
                    const margin{cid} = {{ top: 0, right: 50, bottom: 30, left: 10 }};
                    const width{cid} = {400 if self.horizontal else 600} - margin{cid}.left - margin{cid}.right;
                    const barHeight{cid} = 30;
                    const height{cid} = data{cid}.length * barHeight{cid}+30;

                    const svg{cid} = d3.select("#bar-chart{cid}")
                                      .append("svg")
                                      .attr("width", width{cid} + margin{cid}.left + margin{cid}.right)
                                      .attr("height", height{cid} + margin{cid}.top + margin{cid}.bottom)
                                      .append("g")
                                      .attr("transform", `translate(${{margin{cid}.left}}, ${{margin{cid}.top}})`);

                    const y{cid} = d3.scaleBand()
                        .domain(data{cid}.map(d => d.title))
                        .range([0, height{cid}])
                        .padding(0.2);

                    const x{cid} = d3.scaleLinear().domain([0, {max_val}])
                        .nice()
                        .range([0, width{cid}]);
                        
                    const colorScale{cid} = d3.scaleLinear()
                    .domain([0, 0.5, 1])
                    .range(["#77dd77", "#ffb347", "#ff6961"]);

                    const formatNumber{cid} = d3.format(".3f"); // 3 decimal places

                    // Draw bars
                    svg{cid}.selectAll(".bar-val")
                        .data(data{cid})
                        .enter()
                        .append("rect")
                        .attr("class", "bar-val")
                        .attr("y", d => y{cid}(d.title))
                        .attr("x", 0)
                        .attr("height", y{cid}.bandwidth())
                        .attr("width", d => x{cid}(d.val))
                        .attr("fill", d => colorScale{cid}(Math.abs(d.val - d.target)));

                    // Add the label (title) right outside the bar
                    svg{cid}.selectAll(".bar-label")
                        .data(data{cid})
                        .enter()
                        .append("text")
                        .attr("class", "bar-label")
                        .attr("x", d => 5) // 5px padding inside the bar
                        .attr("y", d => y{cid}(d.title) + y{cid}.bandwidth() / 2)
                        .attr("dy", ".35em")
                        .text(d => d.title)
                        .attr("fill", "black")
                        .attr("font-size", "12px")
                        .attr("text-anchor", "start");

                    // Axes
                    svg{cid}.append("g")
                        .call(d3.axisLeft(y{cid}).tickFormat("")); // no labels on y axis

                    svg{cid}.append("g")
                        .attr("transform", `translate(0, ${{height{cid}}})`)
                        .call(d3.axisBottom(x{cid}).tickFormat(d => (d / {max_val}).toFixed(1)));
                </script>
            """
        else:
            self.contents += f"""
                <script>
                    const data{cid} = {bar_json};
                    const margin{cid} = {{ top: 20, right: 80, bottom: 80, left: 30 }};
                    const width{cid} = {400 if self.horizontal else 600} - margin{cid}.left - margin{cid}.right;
                    const height{cid} = 200 - margin{cid}.top - margin{cid}.bottom;
    
                    const svg{cid} = d3.select("#bar-chart{cid}")
                                      .append("svg")
                                      .attr("width", width{cid} + margin{cid}.left + margin{cid}.right)
                                      .attr("height", height{cid} + margin{cid}.top + margin{cid}.bottom)
                                      .append("g")
                                      .attr("transform", `translate(${{margin{cid}.left}}, ${{margin{cid}.top}})`);
    
                    const x{cid} = d3.scaleBand()
                        .domain(data{cid}.map(d => d.title))
                        .range([0, width{cid}])
                        .padding(0.2);
    
                    svg{cid}.append("g")
                       .attr("transform", `translate(0, ${{height{cid}}})`)
                       .call(d3.axisBottom(x{cid}))
                       .selectAll("text")
                       .attr("transform", "translate(0,10) rotate(30) translate(-10,-10)")
                       .style("text-anchor", "start");
    
                    const y{cid} = d3.scaleLinear()
                        .domain([0, {max_val}])
                        .nice()
                        .range([height{cid}, 0]);
    
                    svg{cid}.append("g")
                       .call(d3.axisLeft(y{cid}));
    
                    const colorScale{cid} = d3.scaleLinear()
                        .domain([0, 0.5, 1])
                        .range(["#77dd77", "#ffb347", "#ff6961"]);
    
                    svg{cid}.selectAll(".bar-val")
                       .data(data{cid})
                       .enter()
                       .append("rect")
                       .attr("class", "bar-val")
                       .attr("x", d => x{cid}(d.title))
                       .attr("y", d => y{cid}(d.val))
                       .attr("width", x{cid}.bandwidth())
                       .attr("height", d => height{cid} - y{cid}(d.val))
                       .attr("fill", d => colorScale{cid}(Math.abs(d.val - d.target)));
                </script>
            """
        if not self.distributions:
            self.contents += "\n</details>\n"

        self.bars = list()

    def quote(self, text, keywords=()):
        for keyword in keywords:
            text = text.replace(
                keyword,
                f'<span class="text-secondary font-weight-bold">{keyword}</span>',
            )
        if self.legend:
            self.contents += f"<i>{text}</i>"
        return self

    def result(self, title, val, target, units=""):
        if not self.legend:
            assert (
                units
            ), "All displayed quantities must have non-empty inferred units when visualizing values with Html and parameter `legend=False`."
        if title == "Value:":
            title = ""
        if abs(val - target) < 0.25:
            background = "#77dd77"  # pastel green
            symbol = "&#x2714;"  # checkmark
            hint = "Near ideal value (does not necessarily mean fair)"
        elif abs(val - target) < 0.75:
            background = "#ffb347"  # pastel orange
            symbol = "?"  # question mark
            hint = "Not ideal / ideal value unknown"
        else:
            background = "#ff6961"  # pastel red
            symbol = "X"  # cross
            hint = "Far from ideal"

        stamp_html = ""
        if symbol:
            stamp_html = (
                f'<span style="'
                f"color:black; "
                f"background:{background}; "
                f"display: inline-block; "
                f"border-radius: 8px;"
                f"border: 1px solid black; "
                f"padding: 2px 6px; padding-bottom: 10px;"
                f"margin-right: 8px; "
                f"font-weight: bold; "
                f"font-size: 2em; "
                f"text-align: center; "
                f"vertical-align: middle; "
                f"width: 1.5em; height: 1.5em; "
                f"box-shadow: 1px 1px 5px rgba(0, 0, 0, 0.2);"
                f'" title="{hint}">{symbol}</span>'
            )

        self.contents += (
            f'<div class="mt-3 alert" style="background-color: white; padding:0px;">'
            f"{stamp_html}<span style='font-size: 1.3em; vertical-align: middle;'>{title} <b>{val:.3f}</b> {units}</span></div>"
        )
        return self

    def first(self):
        return self

    def bold(self, text):
        if self.legend:
            self.contents += f"<br>{text}"
        return self

    def text(self, text):
        if self.legend:
            self.contents += f"{text}<br>"
        return self

    def p(self):
        return self

    def end(self):
        if self.bars:
            self._embed_bars()
        if self.curves:
            self._embed_curves()
        if self.prev_max_level >= 3:
            self.contents += "</div></div>"
        self.contents += "<br><br>"
        return self

    def _create_text(self):
        bootstrap_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>FairBench</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
            <script src="https://d3js.org/d3.v7.min.js"></script>
        </head>
        <body>
            <div {'class="container"' if not self.horizontal else ""}>
                {self.contents}
            </div>
        </body>
        </html>
        """
        return bootstrap_html

    def show(self):
        if self.filename is None:
            return self._create_text()
        with open(f"{self.filename}.html", "w", encoding="utf-8") as temp_file:
            temp_file.write(self._create_text())
            temp_file_path = temp_file.name
        if self.view:
            webbrowser.open(f"{temp_file_path}")
