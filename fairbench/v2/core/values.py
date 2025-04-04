from typing import Optional
import numpy as np

complicated_mode = False


def mismatch(item, keys):
    keys = list(keys)
    ret = f"Key {item} is not one of {list(keys)}.\n"
    similar = [key for key in keys if item in key]
    if similar:
        ret += (
            f"It may be the case that you tried to access one of {similar}.\n"
            "If the dot notation `value.key` is failing due to using invalid Python variable names (e.g., that contain the '&' character), "
            "try the item access notation `value['key']` instead.\n"
        )
    return ret


class Curve:
    def __init__(self, x, y, units: str = ""):
        self.x = np.array(x)
        self.y = np.array(y)
        self.units = units
        for value in x:
            assert not np.isnan(value), f"Cannot store a NaN value in a curve's x-axis"
        for value in y:
            assert not np.isnan(value), f"Cannot store a NaN value in a curve's y-axis"

    def to_dict(self):
        return {
            "x": self.x.tolist(),  # Convert numpy array to list for serialization
            "y": self.y.tolist(),
            "units": self.units,
        }

    def to_grid(self, grid):
        new_x = np.linspace(self.x.min(), self.x.max(), num=grid)
        approx_y = np.interp(new_x, self.x, self.y)
        return Curve(new_x, approx_y, self.units)

    def __str__(self):
        return f"{self.units} curve of {len(self.x)} points"

    @classmethod
    def from_dict(cls, data):
        if "x" not in data or "y" not in data:
            raise ValueError("Dictionary must contain 'x' and 'y' keys.")
        x = np.array(data["x"])  # Convert lists back to numpy arrays
        y = np.array(data["y"])
        units = data.get("units", "")
        return cls(x, y, units)

    def __eq__(self, other):
        if not isinstance(other, Curve):
            return False
        return (
            np.abs(self.x - other.x).sum() == 0
            and np.abs(self.y - other.y).sum() == 0
            and self.units == other.units
        )


class Number:
    def __init__(self, value, units: str = ""):
        self.value = float(value)
        self.units = units

    def __float__(self):
        return self.value

    def to_dict(self):
        return {"value": self.value, "units": self.units}

    def __str__(self):
        return f"{self.value:.3f} {self.units}"

    @classmethod
    def from_dict(cls, data):
        if "x" in data and "y" in data:
            return Curve.from_dict(data)
        if "target" in data:
            return TargetedNumber.from_dict(data)
        return cls(data["value"], data["units"])


class TargetedNumber:
    def __init__(self, value, target, units: str = ""):
        value = float(value)
        target = float(target)
        self.value = value
        self.units = units
        self.target = target

    def __float__(self):
        return self.value

    def __str__(self):
        return f"{self.value:.3f} {self.units} (ideal value {self.target:.3f})"

    def to_dict(self):
        return {"value": self.value, "target": self.target, "units": self.units}

    @classmethod
    def from_dict(cls, data):
        return cls(data["value"], data["target"], data["units"])


class Descriptor:
    def __init__(
        self,
        name,
        role,
        details: Optional[str] = None,
        alias: Optional[str] = None,
        prototype: Optional["Descriptor"] = None,
        preferred_units: Optional[str] = None,
    ):
        self.name = name
        self.role = role
        self.details = name + " " + role if details is None else str(details)
        self.alias = name if alias is None else str(alias)
        self.descriptor = self  # interoperability with methods
        self.prototype = self if prototype is None else prototype
        self.preferred_units = (
            self.prototype.alias if preferred_units is None else str(preferred_units)
        )

    def __str__(self):
        return f"[{self.role}] {self.name}"

    def __eq__(self, other):
        return other.descriptor.name == self.name and other.descriptor.role == self.role

    def __call__(self, value: any = None, depends: list["Value"] = None):
        return Value(value, self, depends, units=self.preferred_units)

    def __repr__(self):
        return (
            self.alias + " [" + self.role + "]"
        )  # self.__str__() + " " + str(hex(id(self)))

    def to_dict(self):
        return {
            "name": self.name,
            "role": self.role,
            "details": self.details,
            "alias": self.alias,
            "preferred_units": self.preferred_units,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data["name"],
            role=data["role"],
            details=data.get("details"),
            alias=data.get("alias"),
            preferred_units=data.get("preferred_units"),
        )


missing_descriptor = Descriptor("unknown", "any role", prototype=None)


class Value:
    def __init__(
        self,
        value: any = None,
        descriptor: Descriptor = missing_descriptor,
        depends: list["Value"] = (),
        units: str = "",  # this is only used if a number is automatically created
    ):
        if (
            value is not None
            and not isinstance(value, TargetedNumber)
            and not isinstance(value, Number)
            and not isinstance(value, Curve)
        ):
            value = Number(value, units)
        elif hasattr(value, "units") and not value.units:
            value.units = units  # TODO: there is a chance that we don't want this applied everywhere, in which case move it to fromework.reduction
        self.value = value
        self.descriptor: Descriptor = descriptor.descriptor
        if depends is None:
            depends = []
        depends = [dep for dep in depends if dep.exists()]
        self.depends = {dep.descriptor.alias: dep for dep in depends}

        if len(depends) != len(self.depends):
            for dep in depends:
                assert dep == self.depends[dep.descriptor.alias], (
                    "A descriptor with the same alias was provided more than once but with different values "
                    f"as a dependency of '{self.descriptor}': {', '.join([str(dep.descriptor.alias) for dep in depends])}\n"
                    "The mis-matching values follow:\n\n"
                    f"{str(dep)}\n\n"
                    f"{str(self.depends[dep.descriptor.alias])}\n\n"
                )

    def __eq__(self, other: "Value"):
        if not isinstance(other, Value):
            return False
        if self.descriptor != other.descriptor:
            return False
        if (self.value is None) != (other.value is None):
            return False
        if self.value is not None and isinstance(self.value, Curve):
            return self.value == other.value
        if self.value is not None and float(self) != float(other):
            return False
        for key in self.depends:
            if key not in other.depends:
                return False
        for key in other.depends:
            if key not in self.depends:
                return False
        for key in self.depends:
            if self.depends[key] != other.depends[key]:
                return False
        return True

    @property
    def units(self):
        assert isinstance(self.value, Number) or isinstance(
            self.value, TargetedNumber
        ), "The value has no units"
        return self.value.units

    @property
    def target(self):
        assert isinstance(self.value, TargetedNumber), "The value has no target"
        return self.value.target

    def __float__(self):
        if self.value is None:
            from fairbench.v2.core import NotComputable

            raise NotComputable("Tried to represent None as a float")
        return float(self.value)

    def _keys(self, role=None, add_to: dict[str, Descriptor] = None):
        if add_to is None:
            add_to = dict()
        for dep in self.depends.values():
            if role is None or dep.descriptor.role == role:
                add_to[dep.descriptor.alias] = dep.descriptor.prototype
            dep._keys(role, add_to)
        return add_to

    def keys(self, role=None):
        return list(self._keys(role).values())

    def values(self, role):
        assert role is not None
        keys = self.keys(role)
        return [self | key for key in keys]

    def single_entry(self):
        if self.value is not None:
            return self
        assert len(self.depends) == 1, (
            "You need to specialize more (e.g., focus on the next mentioned dimension or candidates) to run the requested "
            "operation.\nDetails: it was not possible to retrieve only one value from the following candidates under "
            f"dimension `{self.descriptor}` with alias `{self.descriptor.alias}`: {', '.join(self.depends.keys())}"
        )
        ret = next(iter(self.depends.values())).single_entry()
        item = ret.descriptor
        """return Value(
            value=ret.value,
            descriptor=Descriptor(
                self.descriptor.name + " " + item.name,
                self.descriptor.role + " " + item.role,
                item.details + " of " + self.descriptor.details,
                alias=item.alias,
                ideal_value=item.ideal_value
            ),
            depends=list(ret.depends.values()),
        )"""

        item = Descriptor(
            self.descriptor.name + " " + item.name,
            self.descriptor.role + " " + item.role,
            item.details + " of " + self.descriptor.details,
            alias=item.alias,
        )

        return item(depends=list(ret.depends.values()))

    def flatten(self, to_float=False):
        assert self.value is None, (
            f"Cannot flatten dimension `{self.descriptor}` "
            f"because it already holds a numeric value {float(self.value):.3f}. "
            "Did you mean to work with its `.details` ?"
        )
        assert (
            self.depends
        ), f"There were no retrieved computations to flatten for dimension `{self.descriptor}`"
        if (
            len(self.depends) == 1
            and next(self.depends.values().__iter__()).value is None
        ):
            return next(self.depends.values().__iter__()).flatten(to_float)
        ret = [dep.single_entry() for dep in self.depends.values()]
        if to_float:
            ret = [float(value) for value in ret]
        return ret

    def exists(self) -> bool:
        if self.value is not None:
            return True
        for dep in self.depends.values():
            if dep.exists():
                return True
        return False

    def rebase(self, dep: Descriptor):
        return Value(self.value, dep, list(self.depends.values()))

    def tostring(self, tab="", depth=0, details: bool = False):
        ret = tab + str(self.descriptor.descriptor)
        ret = ret.ljust(40)
        if not self.depends and self.value is None:
            ret += " ---"
            if details:
                ret += f" ({self.descriptor.details})"
            return ret
        if self.value is not None:
            ret += str(self.value)
            depth -= 1
        if details:
            ret += f" ({self.descriptor.details})"
        if depth >= 0:
            for dep in self.depends.values():
                ret += f"\n{dep.tostring(tab+'  ', depth, details)}"
        return ret

    """def serialize(self, depth=0, details: bool = False):
        result = {
            "descriptor": str(self.descriptor.descriptor),
            "value": None if self.value is None else round(self.value, 3),
            "depends": [],
        }
        if not self.depends and self.value is None:
            if details:
                result["details"] = self.descriptor.details
            return result
        if self.value is not None:
            depth -= 1
        if details:
            result["details"] = self.descriptor.details
        if depth >= 0:
            for dep in self.depends.values():
                result["depends"].append(dep.serialize(depth, details))
        return result"""

    def reshape(self, item: Descriptor):
        if isinstance(item, str):
            if item in self.depends:
                item = self.depends[item]
            else:
                # TODO: accelerate this code path in the future
                keys = self._keys()
                assert item in keys, (
                    mismatch(item, keys.keys()) + "Run fb.help(value) for details."
                )
                if item in keys:
                    item = keys[item]
        ret = next(iter(item.descriptor(depends=[self | item]).depends.values()))
        """item = ret.descriptor
        ret.descriptor = Descriptor(
            self.descriptor.name + " " + item.name,
            self.descriptor.role + " " + item.role,
            item.details + " in " + self.descriptor.details,
            alias=self.descriptor.alias + " " + item.alias,
        )"""
        return ret

    def __str__(self):
        return self.tostring()

    def __getitem__(self, item: Descriptor | str) -> "Value":
        if isinstance(item, str):
            if item in self.depends:
                item = self.depends[item]
            else:
                # TODO: accelerate this code path in the future
                keys = self._keys()
                assert item in keys, (
                    mismatch(item, keys.keys()) + "Run `fb.help(value)` for details."
                )
                if item in keys:
                    item = keys[item]
                item = item.descriptor.prototype  # TODO: decide on the prototype
        item = item.descriptor
        item_hasher = item.alias
        if item_hasher in self.depends:
            return self.depends[item_hasher]
        if item_hasher == self.descriptor.alias:
            return self
        """ret = Value(
            None,
            descriptor=Descriptor(
                self.descriptor.name + " " + item.name,
                self.descriptor.role + " " + item.role,
                item.details + " of " + self.descriptor.details,
                alias=item.alias,
                ideal_value=item.ideal_value
            ),
            depends=[dep[item].rebase(dep.descriptor) for dep in self.depends.values()],
        )"""
        # TODO: fix the following
        # depends = [dep[item] for dep in self.depends.values()]
        # if depends and all(dep.value==depends[0].value for dep in depends):
        #    return item(depends=[depends[0][item].rebase(depends[0].descriptor)])#depends[0][item]

        if complicated_mode:
            item = Descriptor(
                self.descriptor.name + " " + item.name,
                self.descriptor.role + " " + item.role,
                item.details + " of " + self.descriptor.details,
                alias=item.alias,
                preferred_units=item.preferred_units,
            )

        ret = item(
            depends=[dep[item].rebase(dep.descriptor) for dep in self.depends.values()]
        )
        return ret

    def __or__(self, other):
        if other is float:
            return float(self)
        return self.__getitem__(other)

    def __and__(self, other):
        return self.reshape(other)

    def __getattr__(self, item):
        if item in dir(self):
            return self.__getattribute__(item)
        return self.__getitem__(item)

    def show(self, env=None, depth=0):
        from fairbench.v2.export.native import format as fmt

        if callable(env):
            env = env()
        if env is not None and hasattr(env, "direct_show"):
            assert (
                depth == 0
            ), f"You cannot specify a depth when showing with env class `{env.__class_.__name__}`"
            return env.direct_show(self)

        return fmt(self, env=env, depth=depth).show()

    def filter(self, *methods):
        if not methods:
            return self
        methods = [
            (
                method()
                if callable(method) and not hasattr(method, "descriptor")
                else method
            )
            for method in methods
        ]
        ret = self
        for method in methods:
            if callable(method) and hasattr(method, "descriptor"):
                # once we have a reduction, get explanation view of internal keys
                ret = ret.explain
                ret = self.descriptor(
                    value=self.value,
                    depends=[method(dep.explain) for dep in ret.depends.values()],
                )
                continue
            ret = method.filter(ret)
        return ret

    def help(self):
        from fairbench.v2.export import help

        return help(self)

    @property
    def explain(self):
        gathered_keys = list()
        for value in self.depends.values():
            gathered_keys.extend(value.depends.keys())
        all = [self | key for key in gathered_keys]
        item = self.descriptor
        roles = set(a.descriptor.role for a in all)
        assert (
            len(roles) == 1
        ), f"While trying to .explain multiple roles were encountered and the operation was aborted: {','.join(roles)}"
        roles = next(roles.__iter__())
        item = Descriptor(
            name=item.name + f" {roles}s",
            role=item.role + f" view",
            details=item.details + f" across {roles}s",
        )
        return item(depends=all)

    @property
    def details(self):
        if self.value is None:
            return self.descriptor(
                depends=[dependency.details for dependency in self.depends.values()]
            )
        item = self.descriptor
        roles = set(a.descriptor.role for a in self.depends.values())
        assert (
            len(roles) == 1
        ), f"While trying to obtain .details multiple roles were encountered and the operation was aborted: {','.join(roles)}"
        roles = next(roles.__iter__())
        item = Descriptor(
            name=item.name + f" {roles}s",
            role=item.role + f" view",
            details=item.details
            + f" across {roles}s"
            + (
                " (NOTE: replace `.details.show()` with '.show(depth=...)' to also see the resulting value)"
                if self.value is not None
                else ""
            ),
        )
        return item(depends=list(self.depends.values()))

    def to_dict(self):
        return {
            "value": self.value.to_dict() if self.value else None,
            "descriptor": self.descriptor.to_dict(),
            "depends": [dep.to_dict() for dep in self.depends.values()],
        }

    @classmethod
    def from_dict(cls, data):
        value = Number.from_dict(data["value"]) if data["value"] else None
        descriptor = Descriptor.from_dict(data["descriptor"])
        depends = [Value.from_dict(dep) for dep in data["depends"]]
        return cls(value=value, descriptor=descriptor, depends=depends)

    """
    def explain(self):
        from fairbench.experimental.export_v2 import help
        from fairbench.experimental.export_v2.formats.ansi import ansi

        help(self)
        print(
            ansi.colorize(
                "#" * 5 + " Numerical details " + "#" * 5, ansi.green + ansi.bold
            ),
            end="",
        )
        return self.show(depth=2)
    """
