from fairbench.v2.export.formats.html import Html
from warnings import warn


class HtmlTable(Html):
    def __init__(
        self,
        view=True,
        filename="temp",
        legend=False,
        transpose=None,
        sideways=True,
        na="   ",
    ):
        super().__init__(filename=filename, view=view)
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

    def colorize(self, text, deviation):
        if deviation < 0.25:
            fg_color = "#28a745"  # Bright green
            tooltip = "Near ideal value (does not necessarily mean fair)"
        elif deviation < 0.75:
            fg_color = "#fd7e14"  # Vibrant orange
            tooltip = "Not ideal / ideal value unknown"
        else:
            fg_color = "#dc3545"  # Strong red
            tooltip = "Far from ideal"

        return (
            f'<td style="color: {fg_color}; width: 50px; height: 50px; text-align: center; vertical-align: middle;" '
            f'data-bs-toggle="tooltip" title="{tooltip}">'
            f"{text}</td>"
        )

    def bar(self, title, val, target, units=""):
        if units == title:
            units = ""
        if units:
            units = "\n(" + units + ")"
        self.bars.append((title, units, val, target))
        return self

    def _embed_bars(self):
        last_title = self.last_title[-1]
        for title, units, val, target in self.bars:
            # Clean row title by removing column header words
            title = " ".join(word for word in title.split() if word != last_title)

            # Create or find the row
            if title not in self.row_names:
                row_num = len(self.row_names)
                self.row_names[title] = row_num
                self.accumulated_bars.append([self.na for _ in self.col_names])
            else:
                row_num = self.row_names[title]

            # Create or find the column
            if last_title not in self.col_names:
                col_num = len(self.col_names)
                self.col_names[last_title] = col_num
                for row in self.accumulated_bars:
                    row.append(self.na)
            else:
                col_num = self.col_names[last_title]

            # Assign value
            assert (
                self.accumulated_bars[row_num][col_num] == self.na
            ), f"Conflict for '{title}' in column '{last_title}'."
            self.accumulated_bars[row_num][col_num] = self.colorize(
                f"{val:.3f}" if val < 1 and val != 0 else str(int(val)),
                abs(val - target),
            )

        self.bars = []
        self.last_supertitle = self.last_title

    def _embed_accumulated_bars(self):
        if self.transpose or (
            self.transpose is None and len(self.row_names) < len(self.col_names)
        ):
            self.transpose = True
            self.accumulated_bars = [
                [self.accumulated_bars[row][col] for row in self.row_names.values()]
                for col in self.col_names.values()
            ]
            self.row_names, self.col_names = self.col_names, self.row_names

        # Calculate full subtable title path
        subtable_title = " ".join(self.last_title[:-1])

        # Append row and column data as raw structures for merging
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

    def _merge_tables_sideways(self):
        if not self.sideways or not self.bar_contents:
            return

        # Initialize merged structures
        merged_rows = {}
        merged_columns = []
        top_headers = []

        # Process each table to align them based on row names
        for table_title, row_names, col_names, table_data in self.bar_contents:
            # Track the subtable title as a header row spanning new columns
            start_col = len(merged_columns)
            col_span = len(col_names)
            top_headers.append((table_title, start_col, col_span))

            # Add new columns to merged columns list
            for col_name in col_names:
                merged_columns.append(col_name)

            # Align rows
            for row_name, row_idx in row_names.items():
                if row_name not in merged_rows:
                    merged_rows[row_name] = [self.na] * len(merged_columns)

                # Expand row to fit new columns if needed
                while len(merged_rows[row_name]) < len(merged_columns):
                    merged_rows[row_name].append(self.na)

                # Fill in values for this table's columns
                for col_name, col_idx in col_names.items():
                    merged_rows[row_name][start_col + col_idx] = table_data[row_idx][
                        col_idx
                    ]

        # Construct the merged HTML table with top headers
        merged_html = "<table class='table'><thead><tr><th></th>"

        # Add top-level headers for each subtable
        last_col = 0
        for header, start_col, col_span in top_headers:
            if start_col > last_col:
                merged_html += f"<th colspan='{start_col - last_col}'></th>"
            merged_html += f"<th colspan='{col_span}'>{header}</th>"
            last_col = start_col + col_span
        if last_col < len(merged_columns):
            merged_html += f"<th colspan='{len(merged_columns) - last_col}'></th>"
        merged_html += "</tr><tr><th></th>"

        # Add column headers
        merged_html += "".join(f"<th scope='col'>{col}</th>" for col in merged_columns)
        merged_html += "</tr></thead><tbody>"

        for row_name, row_values in merged_rows.items():
            merged_html += f"<tr><td>{row_name}</td>"
            merged_html += "".join(row_values)
            merged_html += "</tr>"

        merged_html += "</tbody></table>"
        self.bar_contents = [merged_html]

    def _render_individual_tables(self):
        individual_html = ""
        for table_title, row_names, col_names, table_data in self.bar_contents:
            table = f"<table class='table'><thead><tr><th></th><th colspan='{len(col_names)}'>{table_title}</th></tr><tr><th></th>"
            table += "".join(f"<th scope='col'>{col}</th>" for col in col_names)
            table += "</tr></thead><tbody>"

            for row_name, row_idx in row_names.items():
                table += f"<tr><td>{row_name}</td>"
                table += "".join(table_data[row_idx])
                table += "</tr>"

            table += "</tbody></table>"
            individual_html += table

        self.bar_contents = [individual_html]

    def title(self, text, level=0, link=None):
        if self.bars:
            self._embed_bars()
        if self.curves:
            self._embed_curves()
        if level < self.deepest_title:
            if self.accumulated_bars:
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

        if self.sideways:
            self._merge_tables_sideways()
        else:
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
