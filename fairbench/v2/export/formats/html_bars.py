from fairbench.v2.export.formats.html import Html
from warnings import warn


class HtmlBars(Html):
    def __init__(
        self,
        view=True,
        filename="temp",
        legend=False,
        transpose=None,
        sideways=True,
        na="   ",
        cell_width_px=80,
    ):
        super().__init__(filename=filename, view=view)
        # Core state
        self.accumulated_bars = []
        self.row_names = {}
        self.col_names = {}
        self.last_title = []
        self.last_supertitle = None
        self.bar_contents = []
        self.legend = legend
        self.deepest_title = 0
        self.transpose = transpose
        self.sideways = sideways
        self.last_row_names = None
        self.na = f"<td>{na}</td>"
        self.bars = []
        self.curves = []
        self.contents = ""

        # New rendering support
        self.figures_data = []
        self.cell_width_px = cell_width_px
        self.ratio = 1.0

    # ------------------------------------------------------------------ #
    # Bar accumulation
    # ------------------------------------------------------------------ #
    def bar(self, title, val, target, units=""):
        if units == title:
            units = ""
        if units:
            units = "\n(" + units + ")"
        self.bars.append((title, units, val, target))
        return self

    def _embed_bars(self):
        if not self.last_title:
            return
        last_title = self.last_title[-1]
        for title, units, val, target in self.bars:
            title = " ".join(word for word in title.split() if word != last_title)

            if title not in self.row_names:
                row_num = len(self.row_names)
                self.row_names[title] = row_num
                self.accumulated_bars.append([self.na for _ in self.col_names])
            else:
                row_num = self.row_names[title]

            if last_title not in self.col_names:
                col_num = len(self.col_names)
                self.col_names[last_title] = col_num
                for row in self.accumulated_bars:
                    row.append(self.na)
            else:
                col_num = self.col_names[last_title]

            assert (
                self.accumulated_bars[row_num][col_num] == self.na
            ), f"Conflict for '{title}' in column '{last_title}'."

            # store numeric pair for rendering
            self.accumulated_bars[row_num][col_num] = (float(val), float(target))

        self.bars = []
        self.last_supertitle = self.last_title

    def _embed_accumulated_bars(self):
        if not self.row_names or not self.col_names:
            return

        # Transpose logic
        if self.transpose or (
            self.transpose is None and len(self.row_names) < len(self.col_names)
        ):
            self.transpose = True
            self.accumulated_bars = [
                [self.accumulated_bars[row][col] for row in self.row_names.values()]
                for col in self.col_names.values()
            ]
            self.row_names, self.col_names = self.col_names, self.row_names

        subtable_title = " ".join(self.last_title[:-1])
        self.bar_contents.append(
            (
                subtable_title,
                self.row_names.copy(),
                self.col_names.copy(),
                self.accumulated_bars.copy(),
            )
        )

        self.accumulated_bars = []
        self.row_names = {}
        self.col_names = {}

    # ------------------------------------------------------------------ #
    # Color gradient (green/red depending on deviation)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _color_for(diff, min_val, max_val):
        ratio = abs(diff - min_val) / (max_val - min_val + 1e-9)
        ratio = max(0.0, min(1.0, ratio))

        if diff < 0:
            r = int(128 * (1 - ratio) + 122)
            g = int(255 - 80 * ratio)
            b = 164
        elif diff > 0:
            r = 255
            g = int(164 * (1 - ratio) + 60)
            b = int(164 * (1 - ratio) + 60)
        else:
            r, g, b = 200, 200, 200
        return f"rgb({r},{g},{b})"

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render_individual_tables(self):
        self.figures_data = []

        # Convert accumulated numeric tables into a uniform format
        for table_title, row_names, col_names, table_data in self.bar_contents:
            x_labels = list(col_names.keys())
            y_labels = list(row_names.keys())
            z, text_data = [], []
            for row in table_data:
                z_row, text_row = [], []
                for cell in row:
                    if cell == self.na or cell == "---":
                        z_row.append(None)
                        text_row.append("")
                    elif isinstance(cell, tuple):
                        val, target = cell
                        z_row.append((val, target))
                        text_row.append(str(round(val, 3)))
                    else:
                        try:
                            val = float(str(cell).strip("<td></td>"))
                            z_row.append((val, 0.0))
                            text_row.append(str(round(val, 3)))
                        except Exception:
                            z_row.append(None)
                            text_row.append("")
                z.append(z_row)
                text_data.append(text_row)
            self.figures_data.append((z, text_data, x_labels, y_labels, table_title))

        # Compute color and width scaling
        all_diffs = [
            abs(val - target)
            for z, *_ in self.figures_data
            for row in z
            for cell in row
            if cell is not None
            for val, target in [cell]
        ]
        min_diff, max_diff = (min(all_diffs), max(all_diffs)) if all_diffs else (0, 1)

        all_vals = [
            val
            for z, *_ in self.figures_data
            for row in z
            for cell in row
            if cell is not None
            for val, _ in [cell]
        ]
        max_val = max(all_vals) if all_vals else 1.0

        user_ratio = getattr(self, "ratio", 1.0)

        html_sections = []
        for z, text_data, x_labels, y_labels, title in self.figures_data:
            rows_html = []
            for y, (z_row, text_row) in enumerate(zip(z, text_data)):
                row_label = y_labels[y] if y < len(y_labels) else ""
                cells = []
                for cell, text in zip(z_row, text_row):
                    if cell is None:
                        cells.append(
                            f'<td style="text-align:left;padding:6px;width:{self.cell_width_px}px;"></td>'
                        )
                        continue

                    val, target = cell
                    diff = val - target
                    color = self._color_for(diff, min_diff, max_diff)

                    # width ratio
                    if val <= 1:
                        ratio_val = val * user_ratio
                    else:
                        ratio_val = (val / max_val) * user_ratio
                    ratio_val = max(0.0, min(1.0, ratio_val))

                    bar_html = f"""
                    <div style="position:relative;width:{self.cell_width_px}px;height:14px;">
                        <div style="position:absolute;bottom:0;left:0;height:100%;width:100%;
                                    background:#ccc;border-radius:2px;overflow:hidden;border:1px solid #000;">
                            <div style="position:absolute;top:0;left:0;height:100%;
                                        width:{ratio_val * 100:.1f}%;
                                        background:{color};border-radius:2px 0 0 2px;"></div>
                            <div style="position:relative;z-index:1;width:100%;height:100%;
                                        display:flex;align-items:center;justify-content:center;
                                        font-size:11px;font-family:sans-serif;">
                                {text}
                            </div>
                        </div>
                    </div>
                    """
                    cells.append(
                        f'<td style="border:0;padding:2px;text-align:center;width:{self.cell_width_px}px;">{bar_html}</td>'
                    )
                rows_html.append(
                    f"<tr><th style='text-align:left;'>{row_label}</th>{''.join(cells)}</tr>"
                )

            header_cells = "".join(
                f"<th style='width:{self.cell_width_px}px;'>{x}</th>" for x in x_labels
            )
            html_sections.append(
                f"""
            <div style="margin:20px;">
                <h2 style="font-family:sans-serif;">{title}</h2>
                <table style="border-collapse:collapse;font-family:sans-serif;">
                    <tr><th></th>{header_cells}</tr>
                    {''.join(rows_html)}
                </table>
            </div>
            """
            )

        self.bar_contents = [f"<html><body>{''.join(html_sections)}</body></html>"]

    # ------------------------------------------------------------------ #
    # Section + output management
    # ------------------------------------------------------------------ #
    def title(self, text, level=0, link=None):
        if self.bars:
            self._embed_bars()
        if self.curves:
            self._embed_curves()
        if level < self.deepest_title and self.accumulated_bars:
            self._embed_accumulated_bars()

        self.deepest_title = max(self.deepest_title, level)
        self.last_title = self.last_title[:level] + [text]

        level = 1 + 2 * level
        self.contents += f"<h{level}>{text}</h{level}>\n"
        return self

    def p(self):
        return self

    def bold(self, text):
        self.contents += f"<b>{text}</b>"
        return self

    def curve(self, title, x, y, units):
        super().curve(title, x, y, units)
        if not self.legend:
            warn(f"Curve {title} will not be shown as legend mode is disabled.")
        return self

    def end(self):
        if self.bars:
            self._embed_bars()
        if self.curves:
            self.contents += "\n"
            self._embed_curves()
        if self.accumulated_bars:
            self._embed_accumulated_bars()

        self._render_individual_tables()

        if self.legend:
            self.contents = (
                "".join(self.bar_contents)
                + "\n"
                + self.contents.replace("Computations cover several cases.", "")
            )
        else:
            self.contents = "".join(self.bar_contents)
        return self
