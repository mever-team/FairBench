from fairbench.v2.export.formats.console import Console
from fairbench.v2.export.formats.ansi import ansi


class ConsoleTable(Console):
    def __init__(self, sep: str = " ", end: str = "", legend: bool = True):
        super().__init__()
        self.accumulated_bars = list()
        self.row_names = dict()
        self.col_names = dict()
        self.last_title = ""
        self.sep_token = sep
        self.end_token = end
        self.legend = legend

    def _embed_bars(self):
        for title, units, val, target in self.bars:
            title = " ".join(word for word in title.split() if word != self.last_title)
            # find or create row
            if title not in self.row_names:
                row_num = len(self.row_names)
                self.row_names[title] = row_num
                self.accumulated_bars.append(["---" for _ in self.col_names])
            else:
                row_num = self.row_names[title]
            # find or create col
            if self.last_title not in self.col_names:
                col_num = len(self.col_names)
                self.col_names[self.last_title] = col_num
                for row in self.accumulated_bars:
                    row.append("---")
            else:
                col_num = self.col_names[self.last_title]
            # set value
            assert self.accumulated_bars[row_num][col_num] == "---", (
                f"Two or more conflicting values for showing '{title}' under header '{self.last_title}'. "
                "Maybe reduce the set depth or switch to a different vasualization engine."
            )

            self.accumulated_bars[row_num][col_num] = ansi.colorize(
                f"{val:.3f}", abs(val - target)
            )
        self.bars = list()

    def title(self, text, level=0, link=None):
        if self.bars:
            self._embed_bars()

        if level <= 3:
            self.last_title = text
        else:
            self.last_title += "." + text
        self.level = level
        super().p()
        text = text
        self.contents += ansi.colorize(text.ljust(30), ansi.blue + ansi.bold)
        return self

    def p(self):
        return self

    def bold(self, text):
        self.contents += text
        return self

    def end(self):
        super().end()
        bar_text = ""
        hsep = self.sep_token
        end = self.end_token
        value_just = max(10, max(ansi.visible_length(key) for key in self.col_names))
        title_just = 30
        bar_text += " " * title_just + hsep
        bar_text += hsep.join(ansi.rjust(key, value_just) for key in self.col_names)
        bar_text += end + "\n"
        for row_name, row_num in self.row_names.items():
            bar_text += ansi.ljust(row_name, title_just) + hsep
            bar_text += hsep.join(
                ansi.rjust(self.accumulated_bars[row_num][col_num], value_just)
                for col_num in self.col_names.values()
            )
            bar_text += end + "\n"
        if self.legend:
            bar_text = bar_text + self.contents.replace("|", " ").replace(
                " Computations cover several cases.", ""
            )
        self.contents = bar_text
        return self
