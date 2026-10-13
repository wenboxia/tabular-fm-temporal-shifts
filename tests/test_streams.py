"""第三轮数据流：留出规则、记忆边界与评估边界分离、三个新数据流的结构（修订 3）。"""
import os

import numpy as np
import pytest

from src.data.streams import (GAS_BATCH_SIZES, GRADUAL_SLICE, HOLDOUT_CAP, HOLDOUT_MIN, Stream,
                              carve_stream_holdouts, elec_regimes, load_stream, quarter_rule)
from src.eval.stream_runner import run_stream
from src.memory.context_memory import METHODS, RegimeBalancedDual, make_memory


def _toy(n=400, k=3, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    regime = (np.arange(n) >= n // 2).astype(np.int64)
    return y, regime


class TestQuarterRule:
    def test_cap_quarter_and_minimum(self):
        y = np.array([0] * 400 + [1] * 60 + [2] * 30)
        regime = np.zeros(len(y), dtype=np.int64)
        k = quarter_rule(y, regime, np.ones(len(y), dtype=bool))
        assert k[(0, 0)] == HOLDOUT_CAP            # 400 // 4 = 100 → 上限 50
        assert k[(0, 1)] == 15                     # 60 // 4
        assert (0, 2) not in k                     # 30 // 4 = 7 < 10，不抽
        assert HOLDOUT_MIN == 10

    def test_ineligible_rows_do_not_count(self):
        y = np.zeros(200, dtype=np.int64); regime = np.zeros(200, dtype=np.int64)
        ok = np.zeros(200, dtype=bool); ok[:60] = True
        assert quarter_rule(y, regime, ok)[(0, 0)] == 15


class TestCarve:
    def test_respects_eligibility_and_counts(self):
        y, regime = _toy()
        ok = np.ones(len(y), dtype=bool); ok[100:200] = False
        s = Stream("toy", np.zeros((len(y), 2), np.float32), y, [0, 1, 2], regime, [200], [200], ok,
                   quarter_rule(y, regime, ok))
        stream_idx, hold = carve_stream_holdouts(s, seed=1)
        assert ok[hold].all()
        assert len(np.intersect1d(stream_idx, hold)) == 0 and len(stream_idx) + len(hold) == len(y)
        for (r, c), k in s.holdout_k.items():
            assert ((s.regime[hold] == r) & (s.y[hold] == c)).sum() == k

    def test_deterministic_per_seed(self):
        y, regime = _toy()
        ok = np.ones(len(y), dtype=bool)
        s = Stream("toy", np.zeros((len(y), 2), np.float32), y, [0, 1, 2], regime, [200], [200], ok,
                   quarter_rule(y, regime, ok))
        a = carve_stream_holdouts(s, 3)[1]; b = carve_stream_holdouts(s, 3)[1]; c = carve_stream_holdouts(s, 4)[1]
        assert np.array_equal(a, b) and not np.array_equal(a, c)


class _Recorder(RegimeBalancedDual):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.closed_at = []

    def close_regime(self, rows):
        super().close_regime(rows)
        self.closed_at.append(int(np.max(rows)) + 1)


class TestMemoryBoundaries:
    def _run(self, mem):
        n = 1200
        y = np.arange(n) % 3
        X = np.zeros((n, 2), np.float32)
        regime = (np.arange(n) >= 800).astype(np.int64)
        X_hold = np.zeros((6, 2), np.float32); regime_hold = np.array([0, 0, 0, 1, 1, 1])
        m = _Recorder(100, 0.75, y)
        dummy = lambda Xc, yc, Xq: np.full((len(Xq), 3), 1 / 3)   # noqa: E731
        r = run_stream(m, dummy, X, y, regime, X_hold, regime_hold, [800], stride=50, start=100,
                       ckpt_every=500, mem_boundaries=mem)
        return m, r

    def test_default_closes_at_eval_boundary(self):
        m, r = self._run(None)
        assert m.closed_at == [800] and r.closures.tolist() == [800]

    def test_memory_can_close_earlier_than_eval_boundary(self):
        m, r = self._run([650])
        assert m.closed_at == [650]
        assert 800 in r.ckpt_t.tolist()              # 评估阶段结束时仍然回测
        assert all(int(u) == 0 for t, u in zip(r.ckpt_t, r.ckpt_upto) if t <= 800)


def test_200_row_methods():
    for m in ("dual200", "dual200_rb", "dual200_rb_adwin"):
        assert m in METHODS
    mem = make_memory("dual200_rb", np.zeros((10, 2)), np.zeros(10, dtype=np.int64), 2, 0)
    assert mem.short_cap == 150 and mem.long_cap == 50 and mem.max_context() == 200


def test_elec_regimes_merge_short_edges():
    seg = elec_regimes(943)                       # 1996-05-07 … 1998-12-05
    lengths = np.bincount(seg)
    assert len(lengths) == 10 and lengths.min() >= 30
    assert lengths[0] == 117 and lengths[-1] == 96   # 首段 5/7–8/31，末段 9/1–12/5


def _load(name):
    try:
        return load_stream(name)
    except Exception as e:  # noqa: BLE001  没有缓存且不能联网时跳过
        pytest.skip(f"{name} 不可用：{type(e).__name__}")


class TestRealStreams:
    def test_gas_chronological_batches(self):
        s = _load("gas")
        assert s.X.shape == (13910, 128) and len(s.class_ids) == 6
        assert s.boundaries == [445, 1689, 3275, 3633, 5933, 9546, 10310]
        last = s.y[-GAS_BATCH_SIZES[9]:]
        assert np.bincount(last, minlength=6).tolist() == [600] * 6     # 第 10 批在最后，每类 600
        first = s.y[:GAS_BATCH_SIZES[0]]
        assert np.bincount(first, minlength=6).tolist() == [90, 98, 83, 30, 70, 74]

    def test_electricity_features_and_regimes(self):
        s = _load("electricity")
        assert s.feature_names == ["day", "period", "nswprice", "nswdemand"]
        assert len(s.y) == 45264 and len(s.boundaries) == 9
        assert all(k == 50 for k in s.holdout_k.values()) and len(s.holdout_k) == 20
        day = s.X[:, 0].astype(int); dem = s.X[:, 3]
        for part in (slice(0, 17424), slice(17424, None)):              # 修正后，前后都是周六、周日用电最低
            means = [dem[part][day[part] == d].mean() for d in range(1, 8)]
            assert set(np.argsort(means)[:2] + 1) == {6, 7}

    def test_insects_gradual_two_boundary_arms(self):
        end, start = _load("insects_gradual"), _load("insects_gradual_start")
        n = GRADUAL_SLICE[1] - GRADUAL_SLICE[0]
        assert len(end.y) == len(start.y) == n == 23000
        assert end.boundaries == start.boundaries == [13000]
        assert end.mem_boundaries == [13000] and start.mem_boundaries == [10000]
        assert not end.holdout_ok[10000:13000].any() and end.holdout_ok[:10000].all()
        assert (0, end.class_ids.index(5)) not in end.holdout_k          # 旧概念里第 5 类太少，不抽
        assert np.array_equal(end.X, start.X)


def test_openml_local_cache_checks_fingerprint(tmp_path, monkeypatch):
    import pandas as pd
    import src.data.streams as st
    monkeypatch.setattr(st, "OPENML_CACHE_DIR", str(tmp_path))
    X = np.arange(12, dtype=float).reshape(4, 3); y = np.array(["a", "b", "a", "b"])
    np.savez(tmp_path / "openml_7.npz", X=X, y=y, names=np.array(["f0", "f1", "f2"]))
    A, ys, names = st._openml(7, st._fingerprint(X, pd.Series(y)))      # 离线：只读缓存，不联网
    assert np.array_equal(A, X) and ys.tolist() == y.tolist() and names == ["f0", "f1", "f2"]
    with pytest.raises(RuntimeError, match="预注册"):
        st._openml(7, "0" * 64)
