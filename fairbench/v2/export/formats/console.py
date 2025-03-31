import math

from fairbench.v2.export.formats.ansi import ansi


class Console:
    def __init__(self, view=True, ansiplot=True, width=80, distributions=True):
        self.contents = ""
        self.symbols = {0: "#", 1: "*", 2: "=", 3: "-", 4: "^", 5: '"'}  # sphinx format
        self.level = 0
        self.ansiplot = ansiplot
        self.bars = list()
        self.curves = list()
        self.width = width
        self.view = view
        self.distributions = distributions

    def navigation(self, text, routes: dict):
        return self

    def title(self, text, level=0, link=None):
        if self.bars:
            self._embed_bars()
        if self.curves:
            self._embed_curves()
        self.level = level
        if not text:
            return self
        symbol = self.symbols[level]
        text = symbol * 5 + " " + text + " " + symbol * 5
        self.p()
        self.contents += ansi.colorize(text, ansi.blue + ansi.bold)
        return self.p()

    def curve(self, title, x, y, units):
        if units == title:
            units = ""
        self.curves.append((title, x, y, units))
        return self

    def bar(self, title, val: float, target: float, units: str = ""):
        if units == title:
            units = ""
        assert not math.isnan(val)
        assert not math.isnan(target)
        assert not math.isinf(val)
        assert not math.isinf(target)
        self.bars.append((title, units, val, target))
        return self

    def _embed_curves(self):
        if self.distributions:
            import ansiplot

            canvas = ansiplot.Scaled(40, 10)
            for title, x, y, units in self.curves:
                canvas.plot(x, y, title=title + " " + units)

            text = canvas.text()
            tab = "" if self.level == 0 else (self.level - 1) * "  " + " "
            self.contents += f"\n{tab}  " + f"\n{tab}  ".join(text.split("\n"))
            self.contents += "\n"
        else:
            self.first()
            self.contents += f"Obtained from {len(self.bars)} curves"
            self.p()

        self.curves = list()
        return self

    def _embed_bars(self):
        if not self.distributions:
            self.first()
            self.contents += f"Obtained from {len(self.bars)} values"
            self.p()
            self.bars = list()
            return
        if self.ansiplot:
            import ansiplot

            width_mult = 3 if len(self.bars) < 20 else 2
            canvas = ansiplot.Scaled(width_mult * len(self.bars), 6)

            """common_units = {units for title, units, val, target in self.bars}
            if len(common_units) == 1:
                # TODO: remove once we have proper plots and automatic conversion
                plot = [val for title, units, val, target in self.bars]
                canvas.plot(
                    range(len(plot)),
                    plot,
                    title=self.bars[0][0] + " to " + self.bars[-1][0],
                )
            else:"""
            max_len = max(len(title) for title, units, val, target in self.bars) + 7
            mv = max(val for title, units, val, target in self.bars)
            ch = canvas.height
            num = 1
            if mv == 0:
                mv = 1
            for title, units, val, target in self.bars:
                canvas.bar(
                    num,
                    val,
                    symbol=canvas.current_color() + canvas.palette.block,
                )
                if int(val / mv * ch + 0.25) != int(val / mv * ch):
                    canvas.bar(num, (val, val), symbol=canvas.current_color() + "▆")
                elif int(val / mv * ch + 0.5) != int(val / mv * ch):
                    canvas.bar(num, (val, val), symbol=canvas.current_color() + "▄")
                elif int(val / mv * ch + 0.75) != int(val / mv * ch):
                    canvas.bar(num, (val, val), symbol=canvas.current_color() + "▂")
                canvas.point(
                    num,
                    0,
                    title=title.ljust(max(max_len, 40 - 2 * self.level - 4))
                    + (f"{val:.3f}" if val < 1 and val != 0 else str(int(val)))
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
            max_len = max(len(title) for title, units, val, target in self.bars) + 4
            for title, units, val, target in self.bars:
                self.contents += ("  |" + title).ljust(
                    max(max_len, 40 - 2 * self.level)
                )
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
        self.bars = list()

    def first(self):
        self.text("|")
        return self

    def quote(self, text, keywords=()):
        for keyword in keywords:
            text = text.replace(
                keyword, ansi.colorize(keyword, ansi.white, ansi.reset + ansi.italic)
            )

        endl = self.width - 2 - (0 if self.level == 0 else (self.level - 1) * 2 + 1)
        if ansi.visible_length(text) > endl:
            idx = 0
            while True:
                next = text.find(" ", idx + 1)
                if next == -1 or ansi.visible_length(text[:next]) >= endl:
                    break
                idx = next
            self.contents += ansi.italic + text[: idx + 1] + ansi.reset
            return self.p().first().quote(text[idx + 1 :])

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

    def text(self, text, isinbullet=False):
        endl = self.width - 2 - (0 if self.level == 0 else (self.level - 1) * 2 + 1)
        if isinbullet:
            self.contents += "   "
            endl -= 3
        if ansi.visible_length(text) > endl:
            idx = 0
            while True:
                next = text.find(" ", idx + 1)
                if next == -1 or ansi.visible_length(text[:next]) >= endl:
                    break
                idx = next
            self.contents += text[: idx + 1]
            return (
                self.p()
                .first()
                .text(text[idx + 1 :], isinbullet or text.startswith(" • "))
            )
        self.contents += text
        return self

    def p(self):
        tab = "" if self.level == 0 else (self.level - 1) * "  " + " "
        self.contents += "\n" + tab
        return self

    def end(self):
        if self.bars:
            self._embed_bars()
        if self.curves:
            self._embed_curves()
        self.contents += "\n"
        return self.p()

    def show(self):
        if not self.view:
            return self.contents
        print(self.contents)
