from xmlrpc.client import MAXINT


class Descriptor:
    def __init__(self, name, role, details=""):
        self.name = name
        self.role = role
        self.descriptor = self
        self.details = details

    def __hash__(self):
        return self.name.__hash__()

    def __str__(self):
        return f"[{self.role}] {self.name}"

    def __eq__(self, other):
        return other.descriptor.name == self.name

    def __call__(self, value: any = None, depends: list["Value"] = None):
        return Value(value, self, depends)

    def __repr__(self):
        return self.__str__()+" "+str(hex(id(self)))


missing_descriptor = Descriptor("unknown", "any role")

class Value:
    def __init__(
        self,
        value: any = None,
        descriptor: Descriptor = missing_descriptor,
        depends: list["Value"] = (),
    ):
        self.value = value
        self.descriptor = descriptor.descriptor
        self.depends = (
            {}
            if depends is None
            else {dep.descriptor.name: dep for dep in depends if dep.exists()}
        )

    def __float__(self):
        assert self.value is not None, "Tried to represent None as a float"
        return float(self.value)

    def keys(self, role=None, add_to: set[Descriptor] = None):
        if add_to is None:
            add_to = set()
        for dep in self.depends.values():
            if role is None or dep.descriptor.role == role:
                add_to.add(dep.descriptor)
            dep.keys(role, add_to)
        return add_to

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

    def tostring(self, tab="", depth=0):
        ret = tab + str(self.descriptor.descriptor)
        ret = ret.ljust(35)
        if not self.depends and self.value is None:
            return f"{ret} ---"
        if self.value is not None:
            ret += f" {self.value:.3f}"
            depth -= 1
        if depth >= 0:
            for dep in self.depends.values():
                ret += f"\n{dep.tostring(tab+'  ', depth)}"
        return ret

    def __str__(self):
        return self.tostring()

    def __getitem__(self, item: Descriptor) -> "Value":
        item_name = item.descriptor.name
        if item_name in self.depends:
            return self.depends[item_name]
        return Value(
            None,
            item,
            [dep[item].rebase(dep.descriptor) for dep in self.depends.values()],
        )

    def __or__(self, other):
        if other is float:
            return float(self)
        return self.__getitem__(other)
