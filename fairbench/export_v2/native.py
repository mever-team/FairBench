from fairbench.core_v2.values import Value, TargetedNumber
from fairbench.export_v2.ansi import ansi
import json


def tojson(value: Value, indent="  "):
    return json.dumps(value.serialize(), indent=indent)


def _console(value: Value, depth=0, max_depth=6):
    symbols = {0: "#", 1: "*", 2: "=", 3: "-", 4: "^", 5: '"'}  # sphinx format
    tab = depth * "  "
    title = value.descriptor.name

    def get_ideal():
        if isinstance(value.value, TargetedNumber):
            return value.value.target
        return float(value)+0.5


    if (not value.depends or depth > max_depth) and value.value is not None:
        tab = tab[:-2]
        title = title.ljust(40)
        val = float(value)
        barplot = (
            ""
            if value.descriptor.role == "quantity"
            else ansi.colorize(
                "█" * int(val * 10)
                +("▌" if int(val * 10+0.5)>int(val*10) else "")+ " ",
                abs(val - get_ideal()),
            )
        )
        print(f"{tab}| {title} {val:.3f} {barplot}")
        return
    if depth > max_depth:
        tab = tab[:-2]
        title = title.ljust(40)
        print(f"{tab}| {title} ...")
        return
    symbol = symbols[depth]
    if depth:
        print()
    ansi.print(tab + symbol * 5 + f" {title} " + symbol * 5, ansi.bold + ansi.blue)
    roles = value.descriptor.role.split(" ")
    roles = [role for role in roles if role not in value.descriptor.details]
    roles = " of a ".join(roles)
    details = value.descriptor.details
    if not roles:
        details = tab + "This is " + details + "."
    else:
        details = (
            f"{tab}This {roles}"
            + ansi.colorize(" is ", ansi.white, ansi.reset + ansi.italic)
            + details
            + "."
        )
    details = details.replace(
        " of ", ansi.colorize(" of ", ansi.white, ansi.reset + ansi.italic)
    )
    details = details.replace(
        " for ", ansi.colorize(" for ", ansi.white, ansi.reset + ansi.italic)
    )
    details = ansi.colorize(details, ansi.italic)
    print(details)
    if value.value is not None:
        val = f"{float(value):.3f}"
        val = ansi.colorize(val, abs(float(value) - get_ideal()))
        print(
            ansi.colorize(f"{tab}Value: " + val, ansi.bold)
            + ansi.colorize(
               f" where ideal is {value.value.target:.3f}" if isinstance(value.value, TargetedNumber) else "", ansi.dim
            )
        )
    elif value.depends:
        print(
            tab
            + ansi.colorize("Its value is computed in the following cases.", ansi.bold)
        )
    for dep in value.depends.values():
        _console(dep, depth + 1, max_depth=max_depth)
    if not value.depends and value.value is None:
        print(tab + "| Nothing has been computed.")


def console(value: Value, depth=1):
    _console(value, max_depth=depth)
    print()
