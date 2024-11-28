from fairbench.core_v2.values import Value
import json


def tojson(value: Value, indent="  "):
    return json.dumps(value.serialize(), indent=indent)



def _console(value: Value, depth=0, max_depth=6):
    symbols = {0: "#", 1: "*", 2: "=", 3: "-", 4: "^", 5: "\""}  # sphinx format
    tab = depth*"  "
    title = value.descriptor.name

    if (not value.depends or depth>max_depth) and value.value is not None:
        tab = tab[:-2]
        title = title.ljust(40)
        val = float(value)
        barplot = "" if value.descriptor.role=="quantity" else "█"*int(val*10)
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
    print(tab+symbol*5+f" {title} "+symbol*5)
    roles = value.descriptor.role.split(" ")
    roles = [role for role in roles if role not in value.descriptor.details]
    roles = " ".join(roles)
    if not roles:
        print(tab+"This is "+value.descriptor.details+".")
    else:
        print(tab+"This "+roles+" is "+value.descriptor.details+".")
    if value.value is not None:
        print(f"{tab}Value: {value.value:.3f}")
    else:
        print("It is computed in the following cases.")
    for dep in value.depends.values():
        _console(dep, depth+1, max_depth=max_depth)
    if not value.depends and value.value is None:
        print(tab+"| Nothing has been computed.")

def console(value: Value, depth=0):
    _console(value, max_depth=depth)
    print()