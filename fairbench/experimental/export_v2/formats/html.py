import webbrowser


class Html:
    def __init__(self):
        self.contents = ""
        self.chart_count = 0
        self.bars = []

    def title(self, text, level=0):
        if self.bars:
            self._embed_bars()
            self.bars.clear()
        level = level*2+1
        if level>6:
            level = 6
        if level <= 3:
            self.contents += f'<h{level} class="mt-5 text-primary">{text}</h{level}>'
        else:
            self.contents += f'<h{level} class="mt-3 text-white bg-secondary p-3 rounded">{text}</h{level}>'
        return self

    def bar(self, title, val, target):
        self.bars.append((title, val, target))
        return self

    def _embed_bars(self):
        bar_data = [{"title": t, "val": v, "target": trg} for t, v, trg in self.bars]
        bar_json = str(bar_data).replace("'", '"')
        self.chart_count += 1
        cid = self.chart_count
        self.contents += f"""
            <details>
            <summary class="text-muted">Distribution</summary>
            
            <div id="bar-chart{cid}" class="mt-4"></div>
            </details>
            <script>
                const data{cid} = {bar_json};
                const margin{cid} = {{ top: 20, right: 30, bottom: 40, left: 50 }};
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
                   .attr("transform", "translate(0,10)")
                   .style("text-anchor", "middle");

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

        self.bars.clear()

    def quote(self, text, keywords=()):
        for keyword in keywords:
            text = text.replace(keyword, f'<span class="text-secondary font-weight-bold">{keyword}</span>')
        self.contents += f'<i>{text}</i>'
        return self

    def result(self, title, val, target):
        self.contents += (
            f'<div class="card mt-3">'
            f'  <div class="card-body">'
            f'    <p class="card-text">'
            f'      {title} {val:.3f}<br>'
            f'    </p>'
            f'  </div>'
            f'</div>'
        )
        return self

    def first(self):
        return self

    def bold(self, text):
        self.contents += f'<br>{text}'
        return self

    def text(self, text):
        self.contents += f'<p>{text}</p>'
        return self

    def p(self):
        return self

    def end(self):
        if self.bars:
            self._embed_bars()
            self.bars.clear()
        self.contents += "<br><br>"
        return self

    def display(self):
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
            <div class="container mt-4">
                {self.contents}
            </div>
        </body>
        </html>
        """
        with open("temp.html", "w", encoding="utf-8") as temp_file:
            temp_file.write(bootstrap_html)
            temp_file_path = temp_file.name
        webbrowser.open(f"{temp_file_path}")
