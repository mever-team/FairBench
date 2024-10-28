import fairbench as fb
import numpy as np


def environment():
    for env in ["torch", "tensorflow", "jax", "numpy"]:
        fb.setbackend(env)
        yield env


def test_disparity_metrics():
    for _ in environment():
        men = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        women = np.array([0, 0, 0, 1, 1, 1, 0, 0])
        nonbin = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        sensitive = fb.Fork(men=men, women=women, nonbin=nonbin)
        predictions = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        report = fb.report(
            predictions=predictions,
            sensitive=sensitive,
            metrics=[fb.cvdisparity, fb.eqrep],
        )
        assert report.men.eqrep <= 1
        assert report.men.cvdisparity >= 0


test_disparity_metrics()
