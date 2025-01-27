class PlotlyHeatMap:
    def __init__(self, legend=True):
        # mode: "window" or "html"
        self.legend = legend

        self.accumulated_bars = []
        self.row_names = {}
        self.col_names = {}
        self.last_title = []
        self.last_supertitle = None
        self.bars = []
        self.level = 0

        # Instead of storing figures, store data sets for subplots
        self.figures_data = (
            []
        )  # Each entry will be (z, text_data, x_labels, y_labels, title)

    def _ensure_cell_structure(self):
        for r_idx in range(len(self.accumulated_bars), len(self.row_names)):
            self.accumulated_bars.append(["---"] * len(self.col_names))
        for r_idx, row in enumerate(self.accumulated_bars):
            if len(row) < len(self.col_names):
                self.accumulated_bars[r_idx] += ["---"] * (
                    len(self.col_names) - len(row)
                )

    def _embed_bars(self):
        last_title = self.last_title[-1]
        for title, units, val, target in self.bars:
            title = " ".join(word for word in title.split() if word != last_title)

            # find or create row
            if title not in self.row_names:
                row_num = len(self.row_names)
                self.row_names[title] = row_num
                self._ensure_cell_structure()
                self.accumulated_bars.append(["---" for _ in self.col_names])
            else:
                row_num = self.row_names[title]

            # find or create col
            if last_title not in self.col_names:
                col_num = len(self.col_names)
                self.col_names[last_title] = col_num
                for row in self.accumulated_bars:
                    row.append("---")
            else:
                col_num = self.col_names[last_title]

            self._ensure_cell_structure()

            assert (
                self.accumulated_bars[row_num][col_num] == "---"
            ), f"Two or more conflicting values for '{title}' under header '{last_title}'."
            self.accumulated_bars[row_num][col_num] = (
                round(val, 3) if val < 1 and val != 0 else int(val),
                target,
            )

        self.bars = []
        self.last_supertitle = self.last_title

    def _embed_accumulated_bars(self):
        if not self.accumulated_bars:
            return

        x_labels = list(self.col_names.keys())
        y_labels = list(self.row_names.keys())

        z = []
        text_data = []
        for row in self.accumulated_bars:
            z_row = []
            text_row = []
            for cell in row:
                if cell == "---":
                    z_row.append(None)
                    text_row.append("")
                else:
                    val, target = cell
                    diff = abs(val - target)
                    z_row.append(diff)
                    text_row.append(str(val))
            z.append(z_row)
            text_data.append(text_row)

        title = " ".join(self.last_supertitle[:-1]) if self.last_supertitle else ""

        # Store the dataset for later subplot creation
        self.figures_data.append((z, text_data, x_labels, y_labels, title))

        # Reset data
        self.accumulated_bars = []
        self.row_names = {}
        self.col_names = {}

    def title(self, text, level=0, link=None):
        if self.bars:
            try:
                self._embed_bars()
            except AssertionError:
                self._embed_accumulated_bars()
                self._embed_bars()

        self.last_title = self.last_title[:level]
        self.last_title.append(text)
        self.level = level
        return self

    def bar(self, title, val: float, target: float, units: str = ""):
        if units == title:
            units = ""
        self.bars.append((title, units, val, target))
        return self

    def end(self):
        self._embed_accumulated_bars()
        return self

    def show(self):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if not self.figures_data:
            return self
        num_plots = len(self.figures_data)
        fig = make_subplots(
            rows=1,
            cols=num_plots,
            subplot_titles=[d[4] for d in self.figures_data],
        )
        fig.update_annotations(font=dict(size=20))
        for i, (z, text_data, x_labels, y_labels, title) in enumerate(
            self.figures_data, start=1
        ):
            heatmap = go.Heatmap(
                z=z,
                x=x_labels,
                y=y_labels,
                colorscale="orrd",
                showscale=i == len(self.figures_data),
                colorbar=dict(title="Difference from target"),
                text=text_data,
                texttemplate="%{text}",
                textfont={"color": "black", "size": 10},
            )
            fig.add_trace(heatmap, row=1, col=i)
            fig.update_xaxes(
                row=1,
                col=i,
                tickfont={"color": "black", "size": 16},
                showline=True,  # Enable the outline
                linecolor="black",  # Outline color
                mirror=True,  # Mirror the box (apply to top and bottom)
            )
            fig.update_yaxes(
                row=1,
                col=i,
                tickfont={"color": "black", "size": 16},
                showline=True,  # Enable the outline
                linecolor="black",  # Outline color
                mirror=True,  # Mirror the box (apply to top and bottom)
            )

        fig.update_layout(
            # width=600 * num_plots,  # Scale width based on number of plots
            # height=600,  # Set the height of the figure
            margin=dict(l=10, r=10, t=10, b=10),  # Minimize margins
        )
        fig.show(renderer="browser")

        self.figures_data = []
        return self

    # Dummy methods to maintain compatibility:
    def navigation(self, text, routes: dict):
        return self

    def first(self):
        return self

    def quote(self, text, keywords=()):
        return self

    def result(self, title, val, target, units=""):
        return self

    def bold(self, text):
        return self

    def text(self, text):
        return self

    def p(self):
        return self
