"""
Phase 5.5 Step 6 单元测试：适应 / 遗忘回测工具

三条硬要求各有对应测试：
  1. 留出样本必须从主循环的流里剔除
  2. 回测**不能污染模型状态**（state_hash 前后一致）
  3. 漂移点等时刻要能重映射到剔除后的新坐标
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from src.models.multi_timescale import MultiTimescaleModel
from src.utils.forgetting import (
    ForgettingTracker,
    Holdout,
    backtest_accuracy,
    carve_holdout,
    frozen,
    remap_points,
    state_hash,
)


class _StubSlowPrior:
    def predict_proba(self, X_ctx, y_ctx, X_query):
        n = len(X_query)
        p1 = np.clip(0.5 + 0.4 * np.sign(X_query[:, 0]), 0.01, 0.99)
        return np.stack([1 - p1, p1], axis=1)


def _mk_model(**kw):
    torch.manual_seed(0)
    np.random.seed(0)
    m = MultiTimescaleModel(input_dim=3, **kw)
    m.slow_prior = _StubSlowPrior()
    return m


def _mk_data(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    return X, y


class TestCarveHoldout:

    def test_holdout_rows_are_removed_from_stream(self):
        X, y = _mk_data(1200)
        Xs, ys, kept, hs = carve_holdout(
            X, y, [("A", 0, 600), ("B", 600, 1200)],
            holdout_size=200, context_size=50,
        )
        assert len(Xs) == 1200 - 400
        assert len(kept) == len(Xs)
        assert len(hs) == 2
        # 留出集的每一行都不该出现在剩余流的对应位置
        assert not np.intersect1d(kept, np.arange(400, 600)).size
        assert not np.intersect1d(kept, np.arange(1000, 1200)).size

    def test_holdout_content_matches_source(self):
        X, y = _mk_data(1000)
        _, _, _, hs = carve_holdout(X, y, [("A", 0, 1000)],
                                    holdout_size=300, context_size=50)
        h = hs[0]
        assert np.array_equal(h.X, X[700:1000])
        assert np.array_equal(h.y, y[700:1000])
        assert h.origin_t == 700
        assert len(h) == 300 - 50

    def test_remap_points(self):
        X, y = _mk_data(1000)
        _, _, kept, _ = carve_holdout(X, y, [("A", 0, 500), ("B", 500, 1000)],
                                      holdout_size=100, context_size=30)
        # 原坐标 600 之前被剔除了 100 行（A 的 400-500）
        assert remap_points([600], kept) == [500]
        assert remap_points([0], kept) == [0]

    def test_overlapping_regions_rejected(self):
        X, y = _mk_data(1000)
        with pytest.raises(ValueError, match="重叠"):
            carve_holdout(X, y, [("A", 0, 600), ("B", 500, 1000)],
                          holdout_size=100, context_size=30)

    def test_region_too_small_rejected(self):
        X, y = _mk_data(1000)
        with pytest.raises(ValueError, match="不够留出"):
            carve_holdout(X, y, [("A", 0, 50)], holdout_size=100, context_size=30)

    def test_single_class_context_rejected(self):
        X = np.random.default_rng(0).normal(size=(400, 3)).astype(np.float32)
        y = np.ones(400, dtype=np.int64)       # 全一类
        with pytest.raises(ValueError, match="单一类别"):
            carve_holdout(X, y, [("A", 0, 400)], holdout_size=300, context_size=100)

    def test_holdout_too_short_rejected(self):
        X, y = _mk_data(400)
        with pytest.raises(ValueError, match="不足以留出"):
            Holdout(name="A", X=X[:80], y=y[:80], origin_t=0, context_size=100)


class TestNoContamination:
    """回测绝不能改变模型状态 —— 否则测的不是遗忘，是二次训练。"""

    def test_state_hash_stable_across_backtest(self):
        X, y = _mk_data(800)
        _, _, _, hs = carve_holdout(X, y, [("A", 0, 800)],
                                    holdout_size=300, context_size=100)
        m = _mk_model()
        before = state_hash(m)
        backtest_accuracy(m, hs[0])
        assert state_hash(m) == before

    def test_state_hash_stable_after_real_stepping(self):
        """先跑一段主循环让状态非平凡，再回测，仍不能变。"""
        X, y = _mk_data(800)
        Xs, ys, _, hs = carve_holdout(X, y, [("A", 0, 800)],
                                      holdout_size=300, context_size=100)
        m = _mk_model()
        for t in range(120, 200):
            m.step(Xs[t - 100: t], ys[t - 100: t], Xs[t], float(ys[t]), t=t)
        before = state_hash(m)
        for _ in range(3):
            backtest_accuracy(m, hs[0])
        assert state_hash(m) == before

    def test_state_hash_detects_a_real_change(self):
        """哈希必须真的敏感，否则上面两个测试是空的。"""
        X, y = _mk_data(800)
        m = _mk_model()
        before = state_hash(m)
        m.step(X[0:100], y[0:100], X[100], float(y[100]), t=100)
        assert state_hash(m) != before

    def test_frozen_raises_if_state_mutated(self):
        X, y = _mk_data(400)
        m = _mk_model()
        with pytest.raises(RuntimeError, match="污染"):
            with frozen(m):
                m.fast_corrector.update(X[0], 0.5)   # 故意在冻结期内改状态

    def test_frozen_restores_training_mode(self):
        m = _mk_model()
        m.gated_ensemble.train()
        with frozen(m):
            assert not m.gated_ensemble.training
        assert m.gated_ensemble.training


class TestTracker:

    def _setup(self):
        X, y = _mk_data(1200)
        Xs, ys, kept, hs = carve_holdout(
            X, y, [("early", 0, 600), ("late", 600, 1200)],
            holdout_size=250, context_size=80,
        )
        return Xs, ys, hs

    def test_backtest_accuracy_is_sane(self):
        _, _, hs = self._setup()
        acc = backtest_accuracy(_mk_model(), hs[0])
        assert 0.0 <= acc <= 1.0
        # stub 与标签一致，未训练的 adapter 残差很小 → 应远好于随机
        assert acc > 0.7, acc

    def test_tracker_records_at_checkpoints(self):
        Xs, ys, hs = self._setup()
        m = _mk_model()
        tr = ForgettingTracker(holdouts=hs, every=50)
        for t in range(100, 300):
            m.step(Xs[t - 100: t], ys[t - 100: t], Xs[t], float(ys[t]), t=t)
            tr.maybe_backtest(m, t)
        ts, accs = tr.curve("early")
        assert ts == [100, 150, 200, 250]
        assert len(accs) == 4

    def test_forgetting_uses_best_not_first(self):
        tr = ForgettingTracker(holdouts=[], every=1)
        from src.utils.forgetting import BacktestPoint
        tr.points = [BacktestPoint(t, "A", a, 10)
                     for t, a in zip([1, 2, 3], [0.7, 0.9, 0.6])]
        assert tr.forgetting("A") == pytest.approx(0.3)   # best 0.9 - final 0.6

    def test_forgetting_none_with_single_point(self):
        tr = ForgettingTracker(holdouts=[], every=1)
        from src.utils.forgetting import BacktestPoint
        tr.points = [BacktestPoint(1, "A", 0.7, 10)]
        assert tr.forgetting("A") is None

    def test_summary_shape(self):
        Xs, ys, hs = self._setup()
        m = _mk_model()
        tr = ForgettingTracker(holdouts=hs, every=40)
        for t in range(100, 220):
            m.step(Xs[t - 100: t], ys[t - 100: t], Xs[t], float(ys[t]), t=t)
            tr.maybe_backtest(m, t)
        s = tr.summary()
        assert set(s) == {"early", "late"}
        for v in s.values():
            assert set(v) == {"first", "best", "final", "forgetting", "n"}
