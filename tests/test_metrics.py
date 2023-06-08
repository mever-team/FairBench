import fairbench as fb
import numpy as np


def test_accuracy():
    assert fb.accuracy(np.array([1, 1, 0, 0]), np.array([1, 1, 1, 0])) < 1
    assert fb.accuracy(np.array([1, 1, 0, 0]), np.array([1, 1, 1, 0]), np.array([1, 1, 0, 1])) == 1


def test_tpr():
    assert fb.tpr(np.array([1, 1, 0, 0]), np.array([1, 1, 1, 0])) == 1
    assert fb.tpr(np.array([1, 1, 1, 0]), np.array([1, 1, 0, 0])) < 1
    assert fb.tpr(np.array([1, 1, 1, 0]), np.array([1, 1, 0, 0]), np.array([1, 1, 0, 1])) == 1


def test_tnr():
    assert fb.tnr(np.array([1, 1, 0, 0]), np.array([1, 1, 1, 0])) < 1
    assert fb.tnr(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0])) == 1
    assert fb.tnr(np.array([1, 1, 0, 0]), np.array([1, 1, 1, 0]), np.array([1, 1, 0, 1])) == 1


def test_fpr():
    assert fb.fpr(np.array([1, 1, 0, 0]), np.array([1, 1, 1, 0])) == 0
    assert fb.fpr(np.array([1, 1, 1, 0]), np.array([1, 1, 0, 0])) > 0
    assert fb.fpr(np.array([1, 1, 1, 0]), np.array([1, 1, 0, 0]), np.array([1, 1, 0, 1])) == 0


def test_fnr():
    assert fb.fnr(np.array([1, 1, 0, 0]), np.array([1, 1, 1, 0])) > 0
    assert fb.fnr(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0])) == 0
    assert fb.fnr(np.array([1, 1, 0, 0]), np.array([1, 1, 1, 0]), np.array([1, 1, 0, 1])) == 0
