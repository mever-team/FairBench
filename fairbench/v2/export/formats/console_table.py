from fairbench.v2.export.formats.console import Console
from fairbench.v2.export.formats.ansi import ansi
from warnings import warn


class ConsoleTable(Console):
    def __init__(
        self,
        sep: str = "   ",
        end: str = "",
        legend=False,
        transpose=None,
        sideways=True,
        na="   ",
    ):
        super().__init__()
        self.accumulated_bars = list()
        self.row_names = dict()
        self.col_names = dict()
        self.last_title = list()
        self.last_supertitle = None
        self.sep_token = sep
        self.end_token = end
        self.bar_contents = list()
        self.legend = legend
        self.deepest_title = 0
        self.transpose = transpose
        self.sideways = sideways
        self.last_row_names = None
        self.na = na

    def _embed_bars(self):
        last_title = self.last_title[-1]
        for title, units, val, target in self.bars:
            title = " ".join(word for word in title.split() if word != last_title)
            # find or create row
            if title not in self.row_names:
                row_num = len(self.row_names)
                self.row_names[title] = row_num
                self.accumulated_bars.append([self.na for _ in self.col_names])
            else:
                row_num = self.row_names[title]
            # find or create col
            if last_title not in self.col_names:
                col_num = len(self.col_names)
                self.col_names[last_title] = col_num
                for row in self.accumulated_bars:
                    row.append(self.na)
            else:
                col_num = self.col_names[last_title]
            # set value
            assert self.accumulated_bars[row_num][col_num] == self.na, (
                f"Two or more conflicting values for showing '{title}' under header '{last_title}' "
                f"with all headers being: {', '.join(self.col_names.keys())} "
                "Consider switching to the Console visualization environment instead."
            )
            self.accumulated_bars[row_num][col_num] = ansi.colorize(
                f"{val:.3f}" if val < 1 and val != 0 else str(int(val)),
                abs(val - target),
            )
        self.bars = list()
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
        bar_text = ""
        hsep = self.sep_token
        end = self.end_token
        value_just = max(10, max(ansi.visible_length(key) for key in self.col_names))
        title_just = 30
        if self.last_row_names is None:
            self.last_row_names = ["" for _ in self.row_names]
        assert not self.sideways or len(self.last_row_names) == len(self.row_names), (
            "Sideways ConsoleTable presentation should have the same number of rows in all computed tables. "
            "Consider specializing or constructing `ConsoleTable(sideways=False)` instead."
        )
        has_already_placed_row_names = self.sideways and all(
            name1 == name2 for name1, name2 in zip(self.last_row_names, self.row_names)
        )
        if not has_already_placed_row_names and self.sideways:
            self.last_row_names = ["", "", ""] + [
                ansi.ljust(row_name, title_just) for row_name in self.row_names
            ]
            self.bar_contents.append(
                "\n".join([row + hsep for row in self.last_row_names]) + "\n"
            )
            has_already_placed_row_names = True
        if not has_already_placed_row_names:
            bar_text += " " * title_just + hsep
        bar_text += hsep.join(ansi.rjust(key, value_just) for key in self.col_names)
        bar_text += end + "\n"
        for row_name, row_num in self.row_names.items():
            if not has_already_placed_row_names:
                bar_text += ansi.ljust(row_name, title_just) + hsep
            bar_text += hsep.join(
                ansi.rjust(self.accumulated_bars[row_num][col_num], value_just)
                for col_num in self.col_names.values()
            )
            bar_text += end + "\n"

        bar_text = (
            "\n"
            + ansi.colorize(" ".join(self.last_supertitle[:-1]), ansi.blue + ansi.bold)
            + "\n"
            + bar_text
        )
        self.bar_contents.append(bar_text)
        self.accumulated_bars = list()
        self.last_row_names = self.row_names
        self.row_names = dict()
        self.col_names = dict()

    def title(self, text, level=0, link=None):
        if self.bars:
            self._embed_bars()
        if self.curves:
            self._embed_curves()
        if level < self.deepest_title:
            if self.accumulated_bars:
                self._embed_accumulated_bars()
            self.deepest_title = level
        if level > self.deepest_title:
            self.deepest_title = level
        self.last_title = self.last_title[:level]
        self.last_title.append(text)
        self.level = level
        if level > 0:
            text = " " + text
        super().p()
        self.contents += ansi.colorize(
            text.ljust(30 - 2 * level - (1 if level == 0 else 0)), ansi.blue + ansi.bold
        )
        return self

    def p(self):
        return self

    def bold(self, text):
        self.contents += text
        return self

    def curve(self, title, x, y, units):
        super().curve(title, x, y, units)
        if not self.legend:
            warn(
                f"Curve {title} will not be shown because ConsoleTable was set to legend-less mode."
                "Consider using the `Console` environment or initializing this one with `ConsoleTable(legend=True)`."
            )
        return self

    def end(self):
        # finalize leftover information
        if self.bars:
            self._embed_bars()
        if self.curves:
            self.contents += "\n"
            self._embed_curves()
        if self.accumulated_bars:
            self._embed_accumulated_bars()
        self.contents += "\n"

        # add bar contents
        if self.sideways and self.bar_contents:
            numrows = len(self.bar_contents[0].split("\n"))
            for content in self.bar_contents:
                assert numrows == len(content.split("\n"))
            all_rows = ["" for _ in range(numrows)]
            for content in self.bar_contents:
                rows = content.split("\n")
                max_row_len = max(ansi.visible_length(row) for row in rows)
                sep_token = "" if content == self.bar_contents[0] else self.sep_token
                for i, row in enumerate(rows):
                    if self.end_token:
                        row = row.replace(self.end_token, "")
                    all_rows[i] += sep_token + ansi.ljust(row, max_row_len)
            contents = "\n".join([row + self.end_token for row in all_rows])
        else:
            contents = "".join(self.bar_contents)

        # manage legend
        if self.legend:
            self.contents = contents + self.contents.replace("|", " ").replace(
                " Computations cover several cases.", ""
            )
        else:
            self.contents = contents

        return self
