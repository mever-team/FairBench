import inspect
from fairbench.v1.core.compute.backends import *
from fairbench.v1.core.explanation.error import ExplainableError
from makefun import wraps


def _call_on_branch(_wrapped_method, args, kwargs, branch, transform_args):
    return asprimitive(
        _wrapped_method(
            *(transform_args(arg._branches[branch]) for arg in args),
            **{
                key: (
                    branch if key == "branch" else transform_args(arg._branches[branch])
                )
                for key, arg in kwargs.items()
            },
        )
    )


def _align_branches(_wrapped_method, args, kwargs, Fork, branches):
    args = [
        arg if isinstance(arg, Fork) else Fork(**{branch: arg for branch in branches})
        for arg in args
    ]
    kwargs = {
        key: (
            arg
            if isinstance(arg, Fork)
            else Fork(**{branch: arg for branch in branches})
        )
        for key, arg in kwargs.items()
    }
    argnames = inspect.getfullargspec(_wrapped_method)[0]
    if "branch" not in kwargs and "branch" in argnames:
        kwargs["branch"] = None
    return args, kwargs


def parallel(_wrapped_method):
    @wraps(_wrapped_method)
    def wrapper(*args, **kwargs):
        from fairbench.v1.core import Fork

        try:
            if len(args) == 1 and not kwargs:
                argnames = inspect.getfullargspec(_wrapped_method)[0]
                arg = args[0]
                kwargs = {k: getattr(arg, k) for k in argnames if hasattr(arg, k)}
                args = []
            branches = set(
                [
                    branch
                    for arg in list(args) + list(kwargs.values())
                    if isinstance(arg, Fork)
                    for branch in arg._branches
                ]
            )
            if not branches:
                return asprimitive(
                    _wrapped_method(
                        *(astensor(arg) for arg in args),
                        **{key: astensor(arg) for key, arg in kwargs.items()},
                    )
                )
            args, kwargs = _align_branches(
                _wrapped_method, args, kwargs, Fork, branches
            )
            return Fork(
                **{
                    branch: asprimitive(
                        _call_on_branch(_wrapped_method, args, kwargs, branch, astensor)
                    )
                    for branch in branches
                }
            )
        except ExplainableError as e:
            return e.caught()

    return wrapper


def comparator(_wrapped_method):
    @wraps(_wrapped_method)
    def wrapper(*args, **kwargs):
        from fairbench.v1.core import Fork

        has_fork_of_forks = False
        for arg in args:
            if isinstance(arg, Fork):
                for k, v in arg._branches.items():
                    if isinstance(v, Fork):
                        has_fork_of_forks = True
        for arg in kwargs.values():
            if isinstance(arg, Fork):
                for k, v in arg._branches.items():
                    if isinstance(v, Fork):
                        has_fork_of_forks = True
        if not has_fork_of_forks:
            return _wrapped_method(*args, **kwargs)
        if len(args) == 1 and not kwargs:
            argnames = inspect.getfullargspec(_wrapped_method)[0]
            arg = args[0]
            kwargs = {k: getattr(arg, k) for k in argnames if hasattr(arg, k)}
            args = []
        branches = set(
            [
                branch
                for arg in list(args) + list(kwargs.values())
                if isinstance(arg, Fork)
                for branch in arg._branches
            ]
        )
        if not branches:
            return asprimitive(
                _wrapped_method(
                    *(astensor(arg) for arg in args),
                    **{key: astensor(arg) for key, arg in kwargs.items()},
                )
            )
        args, kwargs = _align_branches(_wrapped_method, args, kwargs, Fork, branches)
        return Fork(
            **{
                branch: asprimitive(
                    _call_on_branch(_wrapped_method, args, kwargs, branch, astensor)
                )
                for branch in branches
            }
        )

    return wrapper


def parallel_primitive(_wrapped_method):
    def tautology(x):
        return x

    @wraps(_wrapped_method)
    def wrapper(*args, **kwargs):
        from fairbench.v1.core import Fork

        if len(args) == 1 and not kwargs:
            argnames = inspect.getfullargspec(_wrapped_method)[0]
            arg = args[0]
            kwargs = {k: getattr(arg, k) for k in argnames if hasattr(arg, k)}
            args = []
        branches = set(
            [
                branch
                for arg in list(args) + list(kwargs.values())
                if isinstance(arg, Fork)
                for branch in arg._branches
            ]
        )
        if not branches:
            try:
                ret = _wrapped_method(*args, **kwargs)
                return ret
            except AttributeError:
                from fairbench.v1 import ExplainableError

                return ExplainableError(
                    f"Cannot call {_wrapped_method.__name__} with arguments {args} {kwargs}"
                )
        args, kwargs = _align_branches(_wrapped_method, args, kwargs, Fork, branches)
        return Fork(
            **{
                branch: _call_on_branch(
                    _wrapped_method, args, kwargs, branch, tautology
                )
                for branch in branches
            }
        )

    return wrapper
