import webbrowser
import json


class Html:
    def __init__(self, horizontal=False, view=True, filename="temp"):
        self.contents = ""
        self.chart_count = 0
        self.bars = []
        self.prev_max_level = 0
        self.routes = dict()
        self.curves = list()
        self.horizontal = horizontal
        self.level = 0
        self.view = view
        self.filename = filename

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
        if level == 3:
            if self.prev_max_level >= level:
                self.contents += "</div></div>"
            else:
                self.contents += "<br>"
            self.contents += f'<div style="width: {400 if self.horizontal else 800}px; float: left;" class="card m-3">'
            self.contents += f'<h{level} class="mt-0 text-white bg-dark p-3 rounded">{text}</h{level}>'

            self.contents += """
            <div class="card-body">
            """
        elif level <= 1:
            self.contents += f'<h{level} class="text-dark">{text}</h{level}>'
        else:
            if level == 5:
                self.contents += "<hr>"
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
        self.chart_count += 1
        cid = self.chart_count

        self.contents += f"""<div id="curve-chart{cid}" class="mt-2"></div>"""

        self.contents += f"""
            <script>
                const curveData{cid} = {curve_json};
                const margin{cid} = {{top: 20, right: 30, bottom: 50, left: 50 }};
                const width{cid} = 400 - margin{cid}.left - margin{cid}.right;
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

            </script>
        """
        self.curves = list()
        return self

    def _embed_bars(self):
        bar_data = [{"title": t, "val": v, "target": trg} for t, v, trg in self.bars]
        bar_json = str(bar_data).replace("'", '"')
        self.chart_count += 1
        cid = self.chart_count
        self.contents += f"""<div id="bar-chart{cid}" class="mt-2"></div>"""
        self.contents += f"""
            <script>
                const data{cid} = {bar_json};
                const margin{cid} = {{ top: 20, right: 80, bottom: 80, left: 30 }};
                const width{cid} = 400 - margin{cid}.left - margin{cid}.right;
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
                    .domain([0, d3.max(data{cid}, d => Math.max(d.val, d.target))])
                    .nice()
                    .range([height{cid}, 0]);

                svg{cid}.append("g")
                   .call(d3.axisLeft(y{cid}));

                const colorScale{cid} = d3.scaleLinear()
                    .domain([0, 0.5, 1])
                    .range(["green", "orange", "red"]);

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

        self.bars = list()

    def quote(self, text, keywords=()):
        for keyword in keywords:
            text = text.replace(
                keyword,
                f'<span class="text-secondary font-weight-bold">{keyword}</span>',
            )
        self.contents += f"<i>{text}</i>"
        return self

    def result(self, title, val, target, units=""):
        if abs(val - target) < 0.25:
            emphasis = "success"
        elif abs(val - target) < 0.75:
            emphasis = "warning"
        else:
            emphasis = "danger"
        self.contents += (
            f'<div class="alert alert-{emphasis} mt-3">{title} {val:.3f} {units}</div>'
        )
        return self

    def first(self):
        return self

    def bold(self, text):
        self.contents += f"<br>{text}"
        return self

    def text(self, text):
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
        with open(f"{self.filename}.html", "w", encoding="utf-8") as temp_file:
            temp_file.write(self._create_text())
            temp_file_path = temp_file.name
        if self.view:
            webbrowser.open(f"{temp_file_path}")
