from fairbench.core.fork.utils import _str_foreign


class Forklike(dict):
    def __init__(self, *args, _role=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._role = _role

    def role(self, _role=None):
        return object.__getattribute__(self, "_role")

    def __getattribute__(self, name):
        if name in dir(Forklike):
            return object.__getattribute__(self, name)
        return self[name]

    def __getitem__(self, item):
        if item in self:
            return super().__getitem__(item)
        return Forklike({k: getattr(v, item) for k, v in self.items()})

    def __str__(self):
        return _str_foreign(self)
