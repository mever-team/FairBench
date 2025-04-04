from collections.abc import Mapping

from fairbench.v1.core.compute import *
from fairbench.v1.core.explanation.error import verify
from fairbench.v1.core.fork.utils import call, _result, _str_foreign
from typing import List


class Fork(Mapping):
    """
    A dictionary-like container that applies element-by-element operations to its contents.
    """

    def __init__(self, *args, _separator="", _role=None, **kwargs):
        self._role = _role
        self._branches = dict()
        # expand keyword arguments
        for arg in args:
            verify(isinstance(arg, dict), "Fork positional arguments can only be dicts")
            for k, v in arg.items():
                verify(k not in kwargs, f"Branch {k} provided multiple times")
                kwargs[k] = v
        # get all keyword arguments while unpacking Categorical data
        for k, v in kwargs.items():
            verify(isinstance(k, str), "Fork branches can only have string names")
            if isinstance(v, dict) and v.__class__.__name__ == "Categorical":
                for attrk, attrv in v.items():
                    name = (
                        str(attrk)
                        if _separator is None
                        else k + _separator + str(attrk)
                    )
                    self._branches[name] = attrv
                continue
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

        ret = dict()
        for k, v in self._branches.items():
            ret[k] = (
                v.__getattribute__(name)
                if isinstance(v, Fork)
                else call(v, "__getattribute__", name)
            )

        return Fork(ret)

    def _extract(self, *args: List[str]):
        """
        Get a view for all the provided arguments, and merge them side-by-side.
        """
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

    def to_dict(self):
        return self.branches()

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
        return Fork(new_branches)

    def intersectional(self, delimiter="&", min_size=1):
        # get branches
        branches = self.branches()
        new_branches = dict()
        for candidates in self.iterate_intersections():
            new_mask = 1
            for branch in candidates:
                new_mask = tobackend(branches[branch]) * new_mask
            if astensor(new_mask).abs().sum() < min_size:
                continue
            new_branches[
                (delimiter.join(candidates)) if len(candidates) > 1 else candidates[0]
            ] = new_mask
        return Fork(new_branches)

    def strict(self):
        branches = self.branches()
        remaining_branches = dict()
        for branch_name, branch_mask in branches.items():
            specification_exists = False
            for other_name, other_mask in branches.items():
                if branch_name == other_name:
                    continue
                branch = tobackend(branch_mask)
                other = tobackend(other_mask)
                both = branch * other
                could_not_be_child = float((both < other).abs().sum().raw) == 0
                if could_not_be_child:
                    specification_exists = True
            if not specification_exists:
                remaining_branches[branch_name] = branch_mask

        return Fork(remaining_branches)

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
        if isinstance(name, list):
            return self._extract(*name)
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
        from fairbench.v1 import Explainable

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
        if override is not None and not isinstance(override, (dict, Fork)):
            return override
        items = (self.branches() if override is None else override).items()
        if any(isinstance(v, dict) for k, v in items):
            html = "".join(
                f'<div style="display: inline-block; float: left;">'
                f"<h3>{k}</h3>{self.__repr_html__(v)}</div>"
                for k, v in items
            )
            return html
        html = (
            "<table>"
            + "".join(
                f"<tr><td><strong>{k}</strong></td><td>{self.__repr_html__(v) if isinstance(v, dict) else asprimitive(v)}</td></tr>"
                for k, v in items
            )
            + "</table>"
        )
        return html
