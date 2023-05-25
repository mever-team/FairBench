import fairbench as fb
import numpy as np


def test_fork_generation():
    fork = fb.Fork(a=1)
    assert fork.a == 1
    fork = fb.Fork(a=1, b=2, c=3)
    assert fork.a == 1
    assert fork.b == 2
    assert fork.c == 3


def test_fork_getattr():
    fork = fb.Fork(a=np.array([1, 2, 3]), b=np.array([2, 3, 4]), c=np.array([3, 4, 5]))
    sums = fork.sum()
    assert sums.a == 6
    assert sums.b == 9
    assert sums.c == 12


def test_fork_of_dicts():
    fork = fb.Fork(a={"x": 1, "y": 2}, b={"x": 2, "y": 4})
    fork["z"] = fork.x + fork.y
    del fork["x"]
    del fork["y"]
    assert len(fork.a) == 1
    assert len(fork.b) == 1
    assert fork.a["z"] == 3
    assert fork.b["z"] == 6


def test_fork_of_forks():
    fork = fb.Fork(a=fb.Fork(x=1, y=2), b=fb.Fork(x=2, y=4))
    assert isinstance(fork.a, fb.Fork)
    assert isinstance(fork.b, fb.Fork)
    assert isinstance(fork.x, fb.Fork)
    assert isinstance(fork.y, fb.Fork)
    assert fork.x.a == fork.a.x
    assert fork.x.b == fork.b.x
    assert fork.y.a == fork.a.y
    assert fork.y.b == fork.b.y
