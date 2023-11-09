import inspect
from fairbench.forks.computations.client_model import client
from fairbench.forks.computations.backends import *
from makefun import wraps


def parallel(_wrapped_method):
    if len(inspect.getfullargspec(_wrapped_method)[0]) <= 1:
        raise Exception(
            "To avoid ambiguity, the @parallel decorator can be applied only to methods with at least"
            "two arguments."
        )

    @wraps(_wrapped_method)
    def wrapper(*args, **kwargs):
        if len(args) == 1 and not kwargs:
            argnames = inspect.getfullargspec(_wrapped_method)[0]
            arg = args[0]
            kwargs = {k: getattr(arg, k) for k in argnames if hasattr(arg, k)}
            args = []
        _client = client()
        from fairbench.forks.fork import Fork
        branches = set(
            [
                branch
                for arg in list(args) + list(kwargs.values())
                if isinstance(arg, Fork)
                for branch in arg._branches
            ]
        )
        if not branches:
            return fromtensor(
                _wrapped_method(
                    *(astensor(arg) for arg in args),
                    **{key: astensor(arg) for key, arg in kwargs.items()},
                )
            )
        args = [
            arg
            if isinstance(arg, Fork)
            else Fork(**{branch: arg for branch in branches})
            for arg in args
        ]
        kwargs = {
            key: arg
            if isinstance(arg, Fork)
            else Fork(**{branch: arg for branch in branches})
            for key, arg in kwargs.items()
        }
        try:
            argnames = inspect.getfullargspec(_wrapped_method)[0]
            if "branch" not in kwargs and "branch" in argnames:
                kwargs["branch"] = None
            submitted = {
                branch: _client.submit(
                    fromtensor,
                    _client.submit(
                        _wrapped_method,
                        *(
                            _client.submit(
                                astensor,
                                arg._branches[branch],
                                workers=branch,
                                allow_other_workers=True,
                                pure=False,
                            )
                            for arg in args
                        ),
                        **{
                            key: branch
                            if key == "branch"
                            else _client.submit(
                                astensor,
                                arg._branches[branch],
                                workers=branch,
                                allow_other_workers=True,
                                pure=False,
                            )
                            for key, arg in kwargs.items()
                        },
                        workers=branch,
                        allow_other_workers=True,
                        pure=False,
                    ),
                    workers=branch,
                    allow_other_workers=True,
                    pure=False,
                )
                for branch in branches
            }
            submitted = {branch: value for branch, value in submitted.items()}
            return Fork(**submitted)
        except KeyError as e:
            raise KeyError(str(e) + " not provided for an input")

    return wrapper



def comparator(_wrapped_method):
    @wraps(_wrapped_method)
    def wrapper(*args, **kwargs):
        _client = client()
        from fairbench.forks.fork import Fork
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

        branches = set(
            [
                branch
                for arg in list(args) + list(kwargs.values())
                if isinstance(arg, Fork)
                for branch in arg._branches
            ]
        )
        if not branches:
            return fromtensor(
                _wrapped_method(
                    *(astensor(arg) for arg in args),
                    **{key: astensor(arg) for key, arg in kwargs.items()},
                )
            )
        args = [
            arg
            if isinstance(arg, Fork)
            else Fork(**{branch: arg for branch in branches})
            for arg in args
        ]
        kwargs = {
            key: arg
            if isinstance(arg, Fork)
            else Fork(**{branch: arg for branch in branches})
            for key, arg in kwargs.items()
        }
        try:
            argnames = inspect.getfullargspec(_wrapped_method)[0]
            if "branch" not in kwargs and "branch" in argnames:
                kwargs["branch"] = None
            submitted = {
                branch: _client.submit(
                    fromtensor,
                    _client.submit(
                        _wrapped_method,
                        *(
                            _client.submit(
                                astensor,
                                arg._branches[branch],
                                workers=branch,
                                allow_other_workers=True,
                                pure=False,
                            )
                            for arg in args
                        ),
                        **{
                            key: branch
                            if key == "branch"
                            else _client.submit(
                                astensor,
                                arg._branches[branch],
                                workers=branch,
                                allow_other_workers=True,
                                pure=False,
                            )
                            for key, arg in kwargs.items()
                        },
                        workers=branch,
                        allow_other_workers=True,
                        pure=False,
                    ),
                    workers=branch,
                    allow_other_workers=True,
                    pure=False,
                )
                for branch in branches
            }
            submitted = {branch: value for branch, value in submitted.items()}
            return Fork(**submitted)
        except KeyError as e:
            raise KeyError(str(e) + " not provided for an input")

    return wrapper


def parallel_primitive(_wrapped_method):
    @wraps(_wrapped_method)
    def wrapper(*args, **kwargs):
        _client = client()
        from fairbench.forks.fork import Fork
        branches = set(
            [
                branch
                for arg in list(args) + list(kwargs.values())
                if isinstance(arg, Fork)
                for branch in arg._branches
            ]
        )
        if not branches:
            return _wrapped_method(
                *((arg) for arg in args),
                **{key: (arg) for key, arg in kwargs.items()},
            )
        args = [
            arg
            if isinstance(arg, Fork)
            else Fork(**{branch: arg for branch in branches})
            for arg in args
        ]
        kwargs = {
            key: arg
            if isinstance(arg, Fork)
            else Fork(**{branch: arg for branch in branches})
            for key, arg in kwargs.items()
        }
        try:
            argnames = inspect.getfullargspec(_wrapped_method)[0]
            if "branch" not in kwargs and "branch" in argnames:
                kwargs["branch"] = None
            submitted = {
                branch: _client.submit(
                    _wrapped_method,
                    *((arg._branches[branch]) for arg in args),
                    **{
                        key: branch if key == "branch" else (arg._branches[branch])
                        for key, arg in kwargs.items()
                    },
                    workers=branch,
                    allow_other_workers=True,
                    pure=False,
                )
                for branch in branches
            }
            submitted = {branch: value for branch, value in submitted.items()}
            return Fork(**submitted)
        except KeyError as e:
            raise KeyError(
                "One of the Modal inputs is missing a "
                + str(e)
                + " branch that other inputs have"
            )

    return wrapper
