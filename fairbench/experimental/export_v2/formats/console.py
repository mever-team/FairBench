from fairbench.experimental.export_v2.formats.ansi import ansi


class Console:
    def __init__(self, ansiplot=False):
        self.contents = ""
        self.symbols = {0: "#", 1: "*", 2: "=", 3: "-", 4: "^", 5: '"'}  # sphinx format
        self.level = 0
        self.ansiplot = ansiplot
        self.bars = []

    def navigation(self, text, routes: dict):
        return self

    def title(self, text, level=0, link=None):
        if self.bars:
            self._embed_bars()
        self.level = level
        symbol = self.symbols[level]
        text = symbol * 5 + " " + text + " " + symbol * 5
        self.p()
        self.contents += ansi.colorize(text, ansi.blue + ansi.bold)
        return self.p()

    def bar(self, title, val: float, target: float, units: str = ""):
        if units == title:
            units = ""
        self.bars.append((title, units, val, target))
        return self

    def _embed_bars(self):
        if self.ansiplot:
            import ansiplot

            canvas = ansiplot.Scaled(3 * len(self.bars), 8)
            num = 1
            for title, units, val, target in self.bars:
                canvas.bar(
                    num,
                    val,
                    title=title.ljust(40 - 2 * self.level - 4)
                    + (f"{val:.3f}" if val <= 1 else str(int(val)))
                    + " "
                    + units,
                )
                num += 1
            canvas.bar(0, 0, symbol=canvas.palette.yaxis)
            text = canvas.text()
            tab = "" if self.level == 0 else (self.level - 1) * "  " + " "
            self.contents += f"\n{tab}  " + f"\n{tab}  ".join(text.split("\n"))
            self.p()
        else:
            for title, units, val, target in self.bars:
                self.contents += ("  |" + title).ljust(40 - 2 * self.level)
                if val > 1:
                    self.contents += str(int(val)) + " " + units
                else:
                    self.contents += ansi.colorize(
                        f"{val:.3f} {units.ljust(len(units)//4*4+4)} "
                        + "█" * int(val * 10)
                        + ("▌" if int(val * 10 + 0.5) > int(val * 10) else "")
                        + " ",
                        abs(val - target),
                    )
                self.p()
        self.bars.clear()

    def first(self):
        self.text("|")
        return self

    def quote(self, text, keywords=()):
        for keyword in keywords:
            text = text.replace(
                keyword, ansi.colorize(keyword, ansi.white, ansi.reset + ansi.italic)
            )
        self.contents += ansi.italic + text + ansi.reset
        return self

    def result(self, title, val, target, units=""):
        self.contents += (
            ansi.bold + title + ansi.colorize(f" {val:.3f} {units}", abs(val - target))
        )
        return self

    def bold(self, text):
        self.contents += ansi.colorize(text, ansi.bold)
        return self

    def text(self, text):
        self.contents += text
        return self

    def p(self):
        tab = "" if self.level == 0 else (self.level - 1) * "  " + " "
        self.contents += "\n" + tab
        return self

    def end(self):
        if self.bars:
            self._embed_bars()
        self.contents += "\n"
        return self

    def show(self):
        print(self.p().contents)
