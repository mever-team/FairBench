import re


class ansi:
    # Reset
    reset = "\033[0m"

    # Text Colors
    black = "\033[30m"
    red = "\033[31m"
    green = "\033[32m"
    yellow = "\033[33m"
    blue = "\033[34m"
    magenta = "\033[35m"
    cyan = "\033[36m"
    white = "\033[37m"

    # Bright Text Colors
    bright_black = "\033[90m"
    bright_red = "\033[91m"
    bright_green = "\033[92m"
    bright_yellow = "\033[93m"
    bright_blue = "\033[94m"
    bright_magenta = "\033[95m"
    bright_cyan = "\033[96m"
    bright_white = "\033[97m"

    # Background Colors
    bg_black = "\033[40m"
    bg_red = "\033[41m"
    bg_green = "\033[42m"
    bg_yellow = "\033[43m"
    bg_blue = "\033[44m"
    bg_magenta = "\033[45m"
    bg_cyan = "\033[46m"
    bg_white = "\033[47m"

    # Bright Background Colors
    bg_bright_black = "\033[100m"
    bg_bright_red = "\033[101m"
    bg_bright_green = "\033[102m"
    bg_bright_yellow = "\033[103m"
    bg_bright_blue = "\033[104m"
    bg_bright_magenta = "\033[105m"
    bg_bright_cyan = "\033[106m"
    bg_bright_white = "\033[107m"

    # Styles
    bold = "\033[1m"
    dim = "\033[2m"
    italic = "\033[3m"
    underline = "\033[4m"
    blink = "\033[5m"
    inverse = "\033[7m"
    hidden = "\033[8m"
    strikethrough = "\033[9m"

    @staticmethod
    def colorize(text, color, reset=None):
        if isinstance(color, float):
            if color < 0.25:
                color = ansi.green
            elif color < 0.75:
                color = ansi.yellow
            else:
                color = ansi.red
        if reset is None:
            reset = ansi.reset
        return f"{color}{text}{reset}"

    @staticmethod
    def print(text, color):
        return print(ansi.colorize(text, color))

    ANSI_COLOR_PATTERN = re.compile(r"\x1b\[[0-9;]*m")

    @staticmethod
    def visible_length(text):
        """Calculate the visible length of a string ignoring ANSI color codes."""
        return len(ansi.ANSI_COLOR_PATTERN.sub("", text))

    @staticmethod
    def ljust(text, width, fillchar=" "):
        """Left-justify a string ignoring ANSI color codes."""
        visible_len = ansi.visible_length(text)
        padding = max(0, width - visible_len)
        return text + fillchar * padding

    @staticmethod
    def rjust(text, width, fillchar=" "):
        """Right-justify a string ignoring ANSI color codes."""
        visible_len = ansi.visible_length(text)
        padding = max(0, width - visible_len)
        return fillchar * padding + text
