from fairbench.experimental.core_v2 import Value, TargetedNumber, Descriptor, Comparison
from fairbench.experimental.export_v2.ansi import ansi
import json

def _generate_details(descriptor: Descriptor):
    roles = descriptor.role.split(" ")
    roles = [role for role in roles if role not in descriptor.details]
    roles = " of a ".join(roles)
    details = descriptor.details
    if not roles:
        details = "This" + ansi.colorize(" is ", ansi.white, ansi.reset + ansi.italic) + details + "."
    else:
        details = (
                f"This {roles}"
                + ansi.colorize(" is ", ansi.white, ansi.reset + ansi.italic)
                + details
                + "."
        )
    details = details.replace(
        " of ", ansi.colorize(" of ", ansi.white, ansi.reset + ansi.italic)
    )
    details = details.replace(
        " in ", ansi.colorize(" in ", ansi.white, ansi.reset + ansi.italic)
    )
    details = details.replace(
        " for ", ansi.colorize(" for ", ansi.white, ansi.reset + ansi.italic)
    )
    details = ansi.colorize(details, ansi.italic)
    return details


def tojson(value: Value, indent="  "):
    return json.dumps(value.serialize(), indent=indent)


def _console(value: Value, depth=0, max_depth=6, tab_delim="", symbol_depth=0):
    symbols = {0: "#", 1: "*", 2: "=", 3: "-", 4: "^", 5: '"'}  # sphinx format
    tab = "" if symbol_depth == 0 else (symbol_depth - 1) * len(tab_delim) * " " + tab_delim
    title = value.descriptor.name

    def get_ideal():
        if isinstance(value.value, TargetedNumber):
            return value.value.target
        return float(value) + 0.5

    if value.value is not None and (depth<max_depth or symbol_depth!=0):
        depth += 1

    if (not value.depends or depth > max_depth or symbol_depth>max_depth) and value.value is not None:
        tab = tab[:-2]
        title = title.ljust(40)
        val = float(value)
        barplot = (
            f"{int(val)}"
            if val > 1  # TODO: make an IntNumber class (like TargetedNumber)
            else ansi.colorize(
                f"{val:.3f}"
                + "█" * int(val * 10)
                + ("▌" if int(val * 10 + 0.5) > int(val * 10) else "")
                + " ",
                abs(val - get_ideal()),
            )
        )
        print(f"{tab}|{title} {barplot}")
        return
    if depth > max_depth:
        tab = tab[:-2]
        title = title.ljust(40)
        print(f"{tab}|{title} ...")
        return
    symbol = symbols[symbol_depth]
    if symbol_depth:
        print()
    ansi.print(tab + symbol * 5 + f" {title} " + symbol * 5, ansi.bold + ansi.blue)
    print(tab + _generate_details(value.descriptor))
    if value.value is not None:
        val = f"{float(value):.3f}"
        val = ansi.colorize(val, abs(float(value) - get_ideal()))
        print(
            ansi.colorize(f"{tab}Value: " + val, ansi.bold)
            + ansi.colorize(
                (
                    f" where ideal is {value.value.target:.3f}"
                    if isinstance(value.value, TargetedNumber)
                    else ""
                ),
                ansi.dim,
            )
        )
    elif value.depends:
        print(
            tab
            + ansi.colorize("A value is computed in the following cases.", ansi.bold)
        )
    for dep in value.depends.values():
        _console(dep, depth, max_depth=max_depth, tab_delim=tab_delim, symbol_depth=symbol_depth+1)
    if not value.depends and value.value is None:
        print(tab + "| Nothing has been computed.")


def console(value: Value, depth=0, tab=" |"):
    # depth=0 gets the minimal details that allow exploration of the next step
    assert isinstance(value, Value), (
        "You did not provide a core.Value. Perhaps you accidentally accessed a property of core.Value instead."
        + "Use the full dict notation (e.g., value['branch'] instead of value.branch to avoid this."
    )
    _console(value, max_depth=depth, tab_delim=tab)
    print()


def help(value: any, details=True):
    if isinstance(value, Comparison) or value==Comparison:
        ansi.print("#" * 5 + " FairBench help " + "#" * 5, ansi.green + ansi.bold)
        ansi.print("Comparison", ansi.bold+ansi.blue)
        print("This is a comparison builder.")
        ansi.print("Usage:", ansi.bold)
        if value==Comparison:
            print("- cmp = Comparison(name)".ljust(27), "Creates a comparison with the given name.")
        print("- cmp.instance(name, value)".ljust(27), "Accumulates a new instance holding the given value.")
        print("- cmp.build()".ljust(27), "Creates a value from accumulated instances and clears cmp.")
        print("- cmp.clear()".ljust(27), "Clears cmp by removing all accumulated instances.")
        print()
        return
    if not isinstance(value, Value):
        if hasattr(value, "descriptor"):
            descriptor = value.descriptor
            if isinstance(descriptor, Descriptor):
                alias = descriptor.name
                ansi.print("#" * 5 + " FairBench help " + "#" * 5, ansi.green + ansi.bold)
                ansi.print(alias, ansi.bold+ansi.blue)
                print(_generate_details(descriptor))
                ansi.print("Usage:", ansi.bold)
                print(f"- value | {alias}".ljust(27), f"Filters a value so that {alias} is the primary focus.")
                if "measure" in descriptor.role:
                    print(f"- {alias}(**kwargs)".ljust(27), "Computes the measure given appropriate arguments.")
                    print(f"- report(measures=[{alias}, ...], ...)")
                if "reduction" in descriptor.role:
                    print(f"- {alias}(values)".ljust(27), "Computes the reduction from an iterable of numeric values.")
                    print(f"- report(reductions=[{alias}, ...], ...)")
                print()
                return
    assert isinstance(value, Value), (
        "You did not provide a fairbench method or value for help. "
        + "Perhaps you accidentally accessed a property of core.Value instead. "
        + "Use the full dict notation (e.g., value['branch'] instead of value.branch to avoid this."
    )
    ansi.print("#"*5 + " FairBench help " + "#"*5, ansi.green+ansi.bold)
    print("Access the following fields of the selected value to explore results:")
    for descriptor in value.keys():
        descriptor = descriptor.prototype
        alias = descriptor.alias
        if " " in alias:
            alias = f"value['{alias}']"
        else:
            alias = "value."+alias
        if details:
            print("-", ansi.colorize(alias.ljust(25), ansi.blue) + " " + _generate_details(descriptor))
        else:
            print("-", ansi.colorize(alias.ljust(25), ansi.blue) + " " + descriptor.role)
    print()