"""
Phase 5.5 Step 8 单元测试：共用恢复目标 + 少犯错误数

针对既有 `adaptation_speed` 的两个问题：
  1. 门槛用各方法**自己**的漂移前均值 → 整体更差的方法门槛更低、反而"恢复更快"
  2. 恢复时刻记在窗口**起点** → 最多提前 window_size-1 步
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.utils.metrics import (
    adaptation_speed,
    errors_avoided,
    recovery_step,
    shared_target_summary,
)


def _stream(pre_acc, post_acc, recover_at, n=1000, drift=300, seed=0):
    """构造一条流：漂移前准确率 pre_acc，漂移后掉到 post_acc，recover_at 步后恢复。"""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    preds = labels.copy()
    for i in range(n):
        acc = pre_acc if i < drift else (post_acc if i < drift + recover_at else pre_acc)
        if rng.random() > acc:
            preds[i] = 1 - preds[i]
    return preds, labels


class TestRecoveryStep:

    def test_recovers_when_accuracy_returns(self):
        preds, labels = _stream(0.95, 0.5, recover_at=200)
        r = recovery_step(preds, labels, drift_point=300, target_acc=0.85,
                          window_size=100, require_consecutive=2)
        assert r is not None and 150 <= r <= 500, r

    def test_returns_none_if_never_recovers(self):
        preds, labels = _stream(0.95, 0.5, recover_at=10_000)
        assert recovery_step(preds, labels, 300, target_acc=0.85,
                             window_size=100) is None

    def test_confirms_at_window_end_not_start(self):
        """记在窗口末尾比记在起点晚，这才是诚实口径。"""
        preds, labels = _stream(0.95, 0.5, recover_at=150)
        end = recovery_step(preds, labels, 300, 0.85, window_size=100,
                            confirm_at_window_end=True)
        start = recovery_step(preds, labels, 300, 0.85, window_size=100,
                              confirm_at_window_end=False)
        assert end is not None and start is not None
        assert end > start

    def test_shared_target_does_not_flatter_a_worse_method(self):
        """核心问题：整体更差的方法在**自适应**门槛下会假装恢复更快。"""
        good_p, good_l = _stream(0.95, 0.5, recover_at=200, seed=1)
        bad_p, bad_l = _stream(0.70, 0.5, recover_at=200, seed=1)

        # 旧口径：各用自己的漂移前均值 × 0.95
        old_good = adaptation_speed(good_p, good_l, [300],
                                    baseline_acc=0.95 * 0.95, window_size=100)[300]
        old_bad = adaptation_speed(bad_p, bad_l, [300],
                                   baseline_acc=0.70 * 0.95, window_size=100)[300]
        assert old_bad is not None and old_good is not None
        assert old_bad <= old_good, (
            "构造前提不成立：差方法在自适应门槛下本应显得不慢"
        )

        # 新口径：共用目标 0.85（取自好方法的水平）
        new_good = recovery_step(good_p, good_l, 300, 0.85, window_size=100)
        new_bad = recovery_step(bad_p, bad_l, 300, 0.85, window_size=100)
        assert new_good is not None
        assert new_bad is None or new_bad > new_good, (
            f"共用门槛下差方法不该更快：good={new_good} bad={new_bad}"
        )

    def test_out_of_range_drift_returns_none(self):
        preds, labels = _stream(0.9, 0.5, 100, n=400)
        assert recovery_step(preds, labels, 9999, 0.8) is None
        assert recovery_step(preds, labels, -5, 0.8) is None

    def test_consecutive_requirement_rejects_a_fluke(self):
        """单个侥幸达标的窗口不算恢复。"""
        labels = np.zeros(600, dtype=int)
        preds = np.ones(600, dtype=int)      # 全错
        # 窗口是从漂移点起的**不重叠**网格 [300,400) [400,500) …，
        # 所以侥幸窗口必须对齐网格才会被评估到
        preds[400:500] = 0                   # 只有一个窗口全对
        one = recovery_step(preds, labels, 300, 0.9, window_size=100,
                            require_consecutive=1)
        two = recovery_step(preds, labels, 300, 0.9, window_size=100,
                            require_consecutive=2)
        assert one is not None
        assert two is None


class TestErrorsAvoided:

    def test_positive_when_better_than_reference(self):
        labels = np.zeros(500, dtype=int)
        ours = labels.copy()
        ref = labels.copy(); ref[100:150] = 1        # 参考多错 50 次
        r = errors_avoided(ours, labels, ref, from_t=100, horizon=100)
        assert r["errors_avoided"] == 50
        assert r["errors_ours"] == 0 and r["errors_ref"] == 50
        assert r["acc_delta_pp"] == pytest.approx(50.0)

    def test_negative_when_worse(self):
        labels = np.zeros(500, dtype=int)
        ours = labels.copy(); ours[100:130] = 1
        ref = labels.copy()
        assert errors_avoided(ours, labels, ref, 100, horizon=100)["errors_avoided"] == -30

    def test_horizon_is_clipped_at_the_end(self):
        labels = np.zeros(200, dtype=int)
        r = errors_avoided(labels, labels, labels, from_t=150, horizon=300)
        assert r["n_steps"] == 50

    def test_out_of_range_returns_none(self):
        labels = np.zeros(200, dtype=int)
        assert errors_avoided(labels, labels, labels, from_t=500, horizon=10) is None

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            errors_avoided(np.zeros(10, dtype=int), np.zeros(10, dtype=int),
                           np.zeros(5, dtype=int), from_t=0)


class TestSharedTargetSummary:

    def test_counts_unrecovered_instead_of_dropping_them(self):
        """未恢复的漂移必须单独计数 —— 静默丢弃会让差方法看起来更快。"""
        preds, labels = _stream(0.95, 0.5, recover_at=10_000, n=1200)
        s = shared_target_summary(preds, labels, [300, 600], target_acc=0.85,
                                  window_size=100)
        assert s["n_not_recovered"] == 2
        assert s["avg_recovery_step"] is None

    def test_errors_avoided_from_alarm_included(self):
        preds, labels = _stream(0.95, 0.5, recover_at=150, n=1000)
        ref, _ = _stream(0.80, 0.5, recover_at=400, n=1000)
        s = shared_target_summary(preds, labels, [300], target_acc=0.85,
                                  reference_predictions=ref,
                                  alarm_times=[320], window_size=100)
        assert len(s["errors_avoided_from_drift"]) == 1
        assert len(s["errors_avoided_from_alarm"]) == 1
        assert s["errors_avoided_from_alarm"][0]["from_t"] == 320

    def test_no_reference_means_no_errors_avoided_keys(self):
        preds, labels = _stream(0.95, 0.5, 150, n=800)
        s = shared_target_summary(preds, labels, [300], target_acc=0.85)
        assert "errors_avoided_from_drift" not in s
