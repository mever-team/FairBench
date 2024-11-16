from makefun import wraps
from collections.abc import Mapping
from fairbench.core.compute.backends import *
from fairbench.core.fork.utils import call, _result, _str_foreign, role
from fairbench.core.fork.forklike import Forklike


class Fork(Mapping):
    def __init__(self, *args, _separator="", _role=None, **branches):
        self._role = _role
        for arg in args:
            if not isinstance(arg, dict):
                raise TypeError(
                    "Forks can only support dicts (holding branch values) as positional arguments"
                )
            for k, v in arg.items():
                if k in branches:
                    raise TypeError(f"Branch {k} provided multiple times")
                branches[k] = v
        self._branches = dict()
        for k, v in branches.items():
            if isinstance(v, dict) and v.__class__.__name__ == "Categorical":
                for k2, v2 in v.items():
                    self._branches[
                        str(k2) if _separator is None else k + _separator + str(k2)
                    ] = v2
            else:
                self._branches[k] = v

    def role(self):
        return object.__getattribute__(self, "_role")

    def __getattribute__(self, name):
        if name in ["_branches", "_repr_html_"] or name in dir(Fork):
            return object.__getattribute__(self, name)
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._branches:
            ret = self._branches[name]
            return _result(ret)

        # def method(*args, **kwargs):
        #    return call(self, name, *args, **kwargs)
        # return method

        return Fork(
            {
                k: (
                    v.__getattribute__(name)
                    if isinstance(v, Fork)
                    else call(v, "__getattribute__", name)
                )
                for k, v in self._branches.items()
            }
        )

    def extract(self, *args):
        import fairbench as fb

        ret = dict()
        for arg in args:
            ret = fb.merge(ret, fb.todict(**{arg: self[arg]}))
        return ret

    def branches(self, branch_names=None, zero_mask=False):
        return {
            branch: (
                _result(value)
                if branch_names is None or not zero_mask or branch in branch_names
                else _result(value) * 0
            )
            for branch, value in self._branches.items()
            if branch_names is None or zero_mask or branch in branch_names
        }

    def withcomplements(self):
        # find missing branch complements
        branches = self.branches()
        new_branches = dict()
        for branch in branches:
            has_complement = False
            for branch2 in branches:
                if (
                    astensor(branches[branch]).abs()
                    - 1
                    + astensor(branches[branch2]).abs()
                ).abs().sum() == 0:
                    has_complement = True
                    break
            if not has_complement:
                new_branches[branch + "'"] = 1 - branches[branch]
        return Fork({**branches, **new_branches})

    def iterate_intersections(self):
        # get branches
        branches = self.branches()
        ids2names = dict(enumerate(branches))
        vec = [0] * len(branches)
        while True:
            # iterate through all different combinations
            vec[-1] += 1
            j = len(vec) - 1
            while j > 0 and vec[j] > 1:
                vec[j] = 0
                vec[j - 1] += 1
                j -= 1
            if j == 0 and vec[0] > 1:
                break
            candidates = [ids2names[i] for i in range(len(vec)) if vec[i] != 0]
            yield candidates

    def relax(self):
        branches = {
            name: asprimitive(branch) for name, branch in self.branches().items()
        }
        from sklearn.naive_bayes import GaussianNB

        new_branches = {}
        for name, branch in branches.items():
            X = np.array([branches[br] for br in branches if br != name]).transpose()
            classifier = GaussianNB()
            classifier.fit(X, branch)
            new_branches[name] = classifier.predict_proba(X)[:, 0].ravel()
            new_branches[name] = new_branches[name] / new_branches[name].max()
            # new_branches[name] = new_branches[name]*(branch.sum()/new_branches[name].sum())
            # print(new_branches[name])
        # print(new_branches)
        return Fork(new_branches)

    def intersectional(self, delimiter="&"):
        # get branches
        branches = self.branches()
        new_branches = dict()
        for candidates in self.iterate_intersections():
            new_mask = 1
            for branch in candidates:
                new_mask = tobackend(branches[branch]) * new_mask
            if astensor(new_mask).abs().sum() == 0:
                continue
            new_branches[
                (delimiter.join(candidates)) if len(candidates) > 1 else candidates[0]
            ] = new_mask
        return Fork(new_branches)

    def __len__(self):
        keys = None
        for k, v in self.branches().items():
            assert isinstance(v, dict)
            v_keys = set(v.keys())
            if keys is None:
                keys = v_keys
            else:
                assert len(v_keys - keys) == 0
                assert len(keys - v_keys) == 0
        return len(keys)

    def __iter__(self):
        keys = None
        for k, v in self.branches().items():
            assert isinstance(v, dict)
            v_keys = set(v.keys())
            if keys is None:
                keys = v_keys
            else:
                assert len(v_keys - keys) == 0
                assert len(keys - v_keys) == 0
        return keys.__iter__()

    def __delitem__(self, name):
        return call(self, "__delitem__", name)

    def __getitem__(self, name):
        if name in self._branches:
            return _result(self._branches[name])
        return call(self, "__getitem__", name)

    def __setitem__(self, name, value):
        return call(self, "__setitem__", name, value)

    def __abs__(self):
        return call(self, "__abs__")

    def __eq__(self, other):
        return call(self, "__eq__", other)

    def __lt__(self, other):
        return call(self, "__lt__", other)

    def __gt__(self, other):
        return call(self, "__gt__", other)

    def __le__(self, other):
        return call(self, "__le__", other)

    def __ge__(self, other):
        return call(self, "__ge__", other)

    def __ne__(self, other):
        return call(self, "__ne__", other)

    def __neg__(self):
        return call(self, "__neg__")

    def __add__(self, other):
        return call(self, "__add__", other)

    def __pow__(self, other):
        return call(self, "__pow__", other)

    def __rpow__(self, other):
        return call(self, "__rpow__", other)

    def __radd__(self, other):
        return call(self, "__add__", other)

    def __sub__(self, other):
        return call(self, "__sub__", other)

    def __rsub__(self, other):
        return call(self, "__rsub__", other)

    def __mul__(self, other):
        return call(self, "__mul__", other)

    def __rmul__(self, other):
        return call(self, "__rmul__", other)

    def __truediv__(self, other):
        return call(self, "__truediv__", other)

    def __rtruediv__(self, other):
        return call(self, "__rtruediv__", other)

    def __floordiv__(self, other):
        return call(self, "__floordiv__", other)

    def __rfloordiv__(self, other):
        return call(self, "__rfloordiv__", other)

    def __or__(self, other):
        return call(self, "__or__", other)

    def __and__(self, other):
        return call(self, "__and__", other)

    def __ror__(self, other):
        return call(self, "__ror__", other)

    def __call__(self, *args, **kwargs):
        from fairbench import Explainable

        return Fork(
            **{
                branch: (
                    value(*args, **kwargs)
                    if not isinstance(value, Explainable)
                    else value
                )
                for branch, value in self._branches.items()
            }
        )

    def __str__(self):
        return _str_foreign(self)

    def __repr__(self):
        # from IPython.display import display_html, HTML
        # display_html(HTML(self.__repr_html__()))
        return super().__repr__()

    def _repr_html_(self):
        return self.__repr_html__()

    def __repr_html__(self, override=None):
        if (
            override is not None
            and not isinstance(override, dict)
            and not isinstance(override, Fork)
        ):
            return override

        complex_contents = any(
            isinstance(v, dict)
            for k, v in (self.branches() if override is None else override).items()
        )
        if complex_contents:
            html = ""
            for k, v in (self.branches() if override is None else override).items():
                html += '<div style="display: inline-block; float: left;">'
                html += "<h3>{}</h3>".format(k)
                html += "{}".format(self.__repr_html__(v))
                html += "</div>"
            return html

        html = "<table>"
        for k, v in (self.branches() if override is None else override).items():
            html += "<tr>"
            html += "<td><strong>{}</strong></td>".format(k)
            if isinstance(v, dict):
                html += "<td>{}</td>".format(self.__repr_html__(v))
            else:
                html += "<td>{}</td>".format(asprimitive(v))
            html += "</tr>"
        html += "</table>"
        return html
