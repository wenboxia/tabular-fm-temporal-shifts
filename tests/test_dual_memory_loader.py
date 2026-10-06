"""
Phase 5.5 Step 7 单元测试：DualMemoryLoader

重点是两个陷阱：
  1. 朴素双记忆在类别均衡流上退化成纯滑窗（context 集合与滑窗完全相同）
  2. 少数类样本被永久钉在长期库里，把过时的 P(y|x) 一直喂给 TabPFN
以及 prequential 安全：yield-then-push，绝不泄漏当前样本。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.data.temporal_loader import DualMemoryLoader, TemporalWindowLoader


def _data(n=600, imbalanced=False, seed=0):
    rng = np.random.default_rng(seed)
    X = np.arange(n, dtype=np.float32).reshape(-1, 1)   # 值 = 下标，便于溯源
    if imbalanced:
        y = np.zeros(n, dtype=np.int64)
        y[: n // 2] = (np.arange(n // 2) % 2)           # 前半均衡
        y[n // 2:] = 1                                  # 后半全是类 1（单类长段）
    else:
        y = (np.arange(n) % 2).astype(np.int64)
    return X, y


class TestBasics:

    def test_capacities_sum_to_context_size(self):
        X, y = _data()
        ld = DualMemoryLoader(X, y, context_size=100, short_ratio=0.75)
        assert ld.short_capacity == 75
        assert ld.long_capacity == 25
        assert ld.short_capacity + ld.long_capacity == 100

    def test_context_size_is_respected(self):
        X, y = _data()
        ld = DualMemoryLoader(X, y, context_size=100, short_ratio=0.5)
        for b in ld:
            assert len(b.X_ctx) == len(b.y_ctx) <= 100

    def test_length_matches_sliding_loader(self):
        X, y = _data()
        a = DualMemoryLoader(X, y, context_size=100, short_ratio=0.5)
        b = TemporalWindowLoader(X, y, context_size=100)
        assert len(a) == len(b)

    def test_bad_ratio_rejected(self):
        X, y = _data()
        for r in (0.0, 1.0, 1.5):
            with pytest.raises(AssertionError):
                DualMemoryLoader(X, y, context_size=100, short_ratio=r)


class TestPrequentialSafety:

    def test_current_sample_never_in_context(self):
        """yield-then-push：预测 t 时 context 里绝不能有 t。"""
        X, y = _data(400)
        for b in DualMemoryLoader(X, y, context_size=80, short_ratio=0.5):
            assert b.t not in b.X_ctx[:, 0].astype(int)

    def test_no_future_samples_in_context(self):
        X, y = _data(400)
        for b in DualMemoryLoader(X, y, context_size=80, short_ratio=0.5):
            assert b.X_ctx[:, 0].max() < b.t

    def test_query_is_the_current_sample(self):
        X, y = _data(300)
        for b in DualMemoryLoader(X, y, context_size=60, short_ratio=0.5):
            assert int(b.X_query[0, 0]) == b.t
            assert int(b.y_query[0]) == int(y[b.t])


class TestNotDegenerate:
    """陷阱 1：必须真的与纯滑窗不同，否则整个实验是空的。"""

    def test_degenerates_to_sliding_while_stream_is_balanced(self):
        """已知性质（实测确认）：类别均衡时双记忆 ≡ 纯滑窗。

        长期库填满后每步进一条出一条，均衡流上进出速率相同，
        长短两库的并集就是最近 budget 条 —— 与滑窗是同一个集合。
        **这不是 bug，是这套机制的适用边界**，解读结果时必须考虑，
        否则会把"双记忆没赢"误读成实现有问题。
        """
        X, y = _data(600, imbalanced=True)   # 前 300 均衡、后 300 单类
        dual = list(DualMemoryLoader(X, y, context_size=100, short_ratio=0.5,
                                     max_age=10_000))
        slide = list(TemporalWindowLoader(X, y, context_size=100))
        same_early = sum(
            set(d.X_ctx[:, 0].astype(int)) == set(s.X_ctx[:, 0].astype(int))
            for d, s in zip(dual, slide) if d.t < 300
        )
        n_early = sum(1 for d in dual if d.t < 300)
        assert same_early == n_early, (
            f"均衡段应与滑窗完全一致，实际只有 {same_early}/{n_early} 步相同"
        )

    def test_differs_from_sliding_once_stream_is_imbalanced(self):
        """单类长段里必须与滑窗不同 —— 否则整个双记忆对照是空的。"""
        X, y = _data(600, imbalanced=True)
        dual = list(DualMemoryLoader(X, y, context_size=100, short_ratio=0.5,
                                     max_age=10_000))
        slide = list(TemporalWindowLoader(X, y, context_size=100))
        # 分歧从 t≈351 开始：不均衡自 t=300 起，需先穿过 50 长的短期库才影响长期库
        pairs = [(d, s) for d, s in zip(dual, slide) if d.t >= 400]
        diffs = sum(
            set(d.X_ctx[:, 0].astype(int)) != set(s.X_ctx[:, 0].astype(int))
            for d, s in pairs
        )
        assert diffs == len(pairs), (
            f"单类长段里只有 {diffs}/{len(pairs)} 步与滑窗不同 —— 退化了"
        )

    def test_divergence_starts_after_short_bank_delay(self):
        """分歧不是立刻发生：不均衡要先穿过短期库才影响长期库的淘汰。"""
        X, y = _data(600, imbalanced=True)
        dual = list(DualMemoryLoader(X, y, context_size=100, short_ratio=0.5,
                                     max_age=10_000))
        slide = list(TemporalWindowLoader(X, y, context_size=100))
        first = next(
            d.t for d, s in zip(dual, slide)
            if set(d.X_ctx[:, 0].astype(int)) != set(s.X_ctx[:, 0].astype(int))
        )
        assert 300 < first <= 300 + 60, f"首次分歧在 t={first}，不在预期区间"

    def test_retains_minority_class_during_single_class_run(self):
        """单类长段里，长期库应仍保有少数类样本 —— 这才是双记忆的价值。"""
        X, y = _data(600, imbalanced=True)
        last = None
        for b in DualMemoryLoader(X, y, context_size=100, short_ratio=0.5,
                                  max_age=10_000):
            last = b
        assert last.t >= 500, "应跑到单类长段深处"
        assert 0 in set(last.y_ctx.tolist()), (
            "单类长段深处 context 里应仍有类 0；没有说明类均衡淘汰没起作用"
        )

    def test_sliding_loses_minority_class_there(self):
        """对照：纯滑窗在同一位置已经完全看不到少数类。"""
        X, y = _data(600, imbalanced=True)
        last = None
        for b in TemporalWindowLoader(X, y, context_size=100):
            last = b
        assert set(last.y_ctx.tolist()) == {1}


class TestAgeCap:
    """陷阱 2：少数类样本不能被永久钉住。"""

    def test_max_age_evicts_stale_entries(self):
        X, y = _data(600, imbalanced=True)
        last = None
        for b in DualMemoryLoader(X, y, context_size=100, short_ratio=0.5,
                                  max_age=120):
            last = b
        ages = last.t - last.X_ctx[:, 0].astype(int)
        assert ages.max() <= 120, f"最老样本年龄 {ages.max()} 超过 max_age=120"

    def test_without_max_age_entries_can_be_pinned(self):
        """不设年龄上限时确实会钉住老样本 —— 说明这个参数不是摆设。"""
        X, y = _data(600, imbalanced=True)
        last = None
        for b in DualMemoryLoader(X, y, context_size=100, short_ratio=0.5,
                                  max_age=None):
            last = b
        ages = last.t - last.X_ctx[:, 0].astype(int)
        assert ages.max() > 120, (
            f"无年龄上限时最老样本才 {ages.max()} 步，测不出钉住效应"
        )
