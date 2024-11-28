class Descriptor:
    def __init__(self, name, role, details: str | None = None, alias: str = None):
        self.name = name
        self.role = role
        self.details = name + " " + role if details is None else details
        self.alias = (
            name if alias is None else alias
        )  # use this to search for the descriptor
        self.descriptor: "Descriptor" = (
            self  # for interoperability with functions that set descriptors
        )

    def __str__(self):
        return f"[{self.role}] {self.name}"

    def __eq__(self, other):
        return other.descriptor.name == self.name

    def __call__(self, value: any = None, depends: list["Value"] = None):
        return Value(value, self, depends)

    def __repr__(self):
        return self.__str__() + " " + str(hex(id(self)))


missing_descriptor = Descriptor("unknown", "any role")


class Value:
    def __init__(
        self,
        value: any = None,
        descriptor: Descriptor = missing_descriptor,
        depends: list["Value"] = (),
    ):
        self.value = value
        self.descriptor: Descriptor = descriptor.descriptor
        self.depends = (
            {}
            if depends is None
            else {dep.descriptor.name: dep for dep in depends if dep.exists()}
        )

    def __float__(self):
        assert self.value is not None, "Tried to represent None as a float"
        return float(self.value)

    def _keys(self, role=None, add_to: dict[str, Descriptor] = None):
        if add_to is None:
            add_to = dict()
        for dep in self.depends.values():
            if role is None or dep.descriptor.role == role:
                add_to[dep.descriptor.alias] = dep.descriptor
            dep._keys(role, add_to)
        return add_to

    def keys(self, role=None):
        return list(self._keys(role).values())

    def single(self):
        if self.value is not None:
            return self
        assert (
            len(self.depends) == 1
        ), f"There were multiple value candidates for dimension `{self.descriptor}`"
        return next(iter(self.depends.values())).single()

    def flatten(self, to_float=False):
        assert (
            self.value is None
        ), f"Cannot flatten a numeric value for dimension {self.descriptor}"
        assert (
            self.depends
        ), f"There were no retrieved computations to flatten for dimension `{self.descriptor}`"
        ret = [dep.single() for dep in self.depends.values()]
        if to_float:
            ret = [float(value) for value in ret]
        return ret

    def exists(self) -> bool:
        if self.value is not None:
            return self.value
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
            ret += f" {self.value:.3f}"
            depth -= 1
        if details:
            ret += f" ({self.descriptor.details})"
        if depth >= 0:
            for dep in self.depends.values():
                ret += f"\n{dep.tostring(tab+'  ', depth, details)}"
        return ret

    def serialize(self, depth=0, details: bool = False):
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
        return result

    def __str__(self):
        return self.tostring()

    def __getitem__(self, item: Descriptor) -> "Value":
        item = item.descriptor
        item_hasher = item.alias
        if item_hasher in self.depends:
            return self.depends[item_hasher]
        if item_hasher == self.descriptor.alias:
            return self
        ret = Value(
            None,
            descriptor=Descriptor(
                self.descriptor.name + " " + item.name,
                self.descriptor.role + " " + item.role,
                item.details + " of " + self.descriptor.details,
                alias=item.alias,
            ),
            depends=[dep[item].rebase(dep.descriptor) for dep in self.depends.values()],
        )
        return ret

    def __or__(self, other):
        if other is float:
            return float(self)
        return self.__getitem__(other)
