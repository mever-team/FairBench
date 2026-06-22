"""
ansi_tee.py — A stdout replacement that:
  - prints normally to the real terminal
  - writes to a file with ANSI color/reset codes converted to HTML <span> tags
"""

import sys
import re

# Maps ANSI color codes to CSS color names
_ANSI_FG = {
    30: "#4a4a4a",   # black → dark gray (pure black is invisible on dark bg)
    31: "#c75f5f",   # red
    32: "#5a9e6f",   # green
    33: "#b8956a",   # yellow → muted amber
    34: "#5f82c7",   # blue
    35: "#9e6aad",   # magenta
    36: "#4f9ea8",   # cyan
    37: "#c0c0c0",   # white → light gray
    # Bright variants
    90: "#707070",   # bright black (dark gray)
    91: "#d47f7f",   # bright red
    92: "#7fbf8f",   # bright green
    93: "#c8aa7a",   # bright yellow
    94: "#7f9fd4",   # bright blue
    95: "#b47fc4",   # bright magenta
    96: "#7fc0ca",   # bright cyan
    97: "#dcdcdc",   # bright white
}

_ANSI_PATTERN = re.compile(r"\x1b\[([0-9;]*)m")


def _ansi_to_html(text: str) -> str:
    """Convert ANSI reset/color escape codes in *text* to HTML <span> tags."""
    result = []
    open_spans = 0  # track how many <span>s we've opened
    pos = 0

    for m in _ANSI_PATTERN.finditer(text):
        # Emit everything before this escape sequence verbatim
        result.append(text[pos:m.start()])
        pos = m.end()

        codes = [int(c) for c in m.group(1).split(";") if c] if m.group(1) else [0]

        for code in codes:
            if code == 0:
                # Reset — close all open spans
                result.append("</span>" * open_spans)
                open_spans = 0
            elif code in _ANSI_FG:
                color = _ANSI_FG[code]
                result.append(f'<span style="color:{color}">')
                open_spans += 1
            # Background and other attributes are ignored

    # Append remaining text and close any dangling spans
    result.append(text[pos:])
    result.append("</span>" * open_spans)
    return "".join(result)


class AnsiTee:
    """
    Drop-in replacement for sys.stdout.

    Writes are passed through to the original stdout unchanged, and also
    written to *file* with ANSI color/reset codes converted to HTML.

    Usage
    -----
        import sys
        from ansi_tee import AnsiTee

        with AnsiTee("output.html") as tee:
            sys.stdout = tee
            print("\x1b[31mRed text\x1b[0m and normal text")
            print("\x1b[32mGreen!\x1b[0m")
            sys.stdout = tee.original

        # or use the context manager which restores stdout automatically:
        with AnsiTee.activate("output.html"):
            print(...)
    """

    def __init__(self, path: str, mode: str = "w", encoding: str = "utf-8"):
        self.original = sys.stdout
        self._file = open(path, mode, encoding=encoding)
        self._file.write(
            "<!DOCTYPE html><html><body>"
            '<pre style="font-family:monospace;background:#222222;color:#c0c0c0;padding:1em;overflow-x:auto;font-size:12px">\n'
        )

    # ------------------------------------------------------------------ #
    # file-like interface                                                  #
    # ------------------------------------------------------------------ #

    def write(self, text: str) -> int:
        self.original.write(text)
        self._file.write(_ansi_to_html(text))
        return len(text)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        self.original.flush()
        self._file.flush()

    def fileno(self):
        return self.original.fileno()

    @property
    def encoding(self):
        return self.original.encoding

    @property
    def errors(self):
        return self.original.errors

    # ------------------------------------------------------------------ #
    # context-manager helpers                                              #
    # ------------------------------------------------------------------ #

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, *_):
        sys.stdout = self.original
        self._file.write("\n</pre></body></html>")
        self._file.close()

    @classmethod
    def activate(cls, path: str, **kwargs):
        """Convenience constructor that can be used directly in `with`."""
        return cls(path, **kwargs)


# ------------------------------------------------------------------ #
# Demo                                                                #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    with AnsiTee.activate("ansi.html"):
        print("Normal text")
        print("\x1b[31mRed\x1b[0m, \x1b[32mGreen\x1b[0m, \x1b[34mBlue\x1b[0m")
        print("\x1b[33mYellow warning\x1b[0m")
        print("\x1b[36mCyan info\x1b[0m — then \x1b[35mmagenta\x1b[0m back to normal")
        print("\x1b[91mBright red error\x1b[0m")
        # Nested / multiple codes in one sequence
        print("Before \x1b[32mgreen \x1b[0m after")
"""
Usage pattern:

from ansiprint import AnsiTee
with AnsiTee.activate("ansi.html"):
"""