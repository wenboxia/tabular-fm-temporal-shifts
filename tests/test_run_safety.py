"""长时间运行的安全网：原子落盘、NaN 立即报错。"""
import os

import numpy as np
import pytest

from src.models.slow_prior import SlowPrior
from src.utils.atomic_io import atomic_savez


def test_atomic_savez_writes_complete_file_and_no_temp(tmp_path):
    p = atomic_savez(str(tmp_path / "run"), predictions=np.arange(5), device=np.array(["cuda"]))
    assert p.endswith("run.npz")
    d = np.load(p, allow_pickle=True)
    assert d["predictions"].tolist() == [0, 1, 2, 3, 4]
    assert str(d["device"][0]) == "cuda"
    assert os.listdir(tmp_path) == ["run.npz"]


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")  # numpy 内部 ZipFile 未关闭，无害
def test_atomic_savez_failure_leaves_no_file(tmp_path):
    class Boom:
        def __array__(self, *a, **k):
            raise RuntimeError("boom")

    with pytest.raises(Exception):
        atomic_savez(str(tmp_path / "bad.npz"), x=Boom())
    assert os.listdir(tmp_path) == []


class _FakeTabPFN:
    def __init__(self, proba):
        self.proba = proba

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return self.proba


def _prior_with(proba):
    sp = SlowPrior.__new__(SlowPrior)
    sp.device, sp.n_estimators, sp._is_fitted = "cuda", 4, False
    sp._model = _FakeTabPFN(proba)
    return sp


def test_slow_prior_raises_on_nan_probabilities():
    sp = _prior_with(np.array([[np.nan, np.nan]]))
    X = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(FloatingPointError):
        sp.predict(X, np.array([0, 1, 0, 1]), X[:1])


def test_slow_prior_passes_finite_probabilities():
    sp = _prior_with(np.array([[0.2, 0.8]]))
    X = np.zeros((4, 3), dtype=np.float32)
    proba, pred = sp.predict(X, np.array([0, 1, 0, 1]), X[:1])
    assert pred.tolist() == [1]
