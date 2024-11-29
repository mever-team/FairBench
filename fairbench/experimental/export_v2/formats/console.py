from fairbench.experimental.export_v2.formats.ansi import ansi


class Console:
    def __init__(self):
        self.contents = ""
        self.symbols = {0: "#", 1: "*", 2: "=", 3: "-", 4: "^", 5: '"'}  # sphinx format
        self.level = 0

    def title(self, text, level=0):
        self.level = level
        symbol = self.symbols[level]
        text = symbol*5 + " " + text + " " + symbol*5
        self.p()
        self.contents += ansi.colorize(text, ansi.blue+ansi.bold)
        return self.p()

    def bar(self, title, val, target):
        self.contents += ("  |"+title).ljust(40-2*self.level)
        if val > 1:
            self.contents += str(int(val))
        else:
            self.contents += ansi.colorize(
                f"{val:.3f} "
                + "█" * int(val * 10)
                + ("▌" if int(val * 10 + 0.5) > int(val * 10) else "")
                + " ",
                abs(val - target),
            )
        return self.p()

    def first(self):
        self.text("|")
        return self

    def quote(self, text, keywords=()):
        for keyword in keywords:
            text = text.replace(keyword, ansi.colorize(keyword, ansi.white, ansi.reset+ansi.italic))
        self.contents += ansi.italic+text+ansi.reset
        return self

    def result(self, title, val, target):
        self.contents += ansi.bold + title + ansi.colorize(f"{val:.3f}", abs(val - target))
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
        self.contents += "\n"
        return self

    def display(self):
        print(self.p().contents)
