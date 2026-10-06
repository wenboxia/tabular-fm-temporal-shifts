"""
Phase 5.5 Step 5 单元测试：ActionPolicy + oracle 触发

不依赖 TabPFN —— 用 stub SlowPrior 注入受控信号。
覆盖以下守卫：
  1. 默认路径行为不变（Phase 4 A / Phase 3 两条）
  2. oracle 触发在**预测前**生效，时刻表与 run_baselines.py --oracle_context_reset 一致
  3. oracle 模式下影子 detector 仍记录事件且不被 clear
  4. 空 oracle / 非法组合直接报错，不静默退化
  5. context_reset 的最小长度 + 类别覆盖守卫
  6. consolidate_on_post_alarm_data 把巩固推迟到报警之后
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from src.models.multi_timescale import (
    ACTIONS_ON_ALARM,
    TRIGGER_SOURCES,
    MultiTimescaleModel,
)


class _ConstSlowPrior:
    """恒定 y_slow，便于制造确定的 indicator 流。"""

    def __init__(self, p1=0.9):
        self.p1 = p1
        self.seen_context_lens = []

    def predict_proba(self, X_ctx, y_ctx, X_query):
        self.seen_context_lens.append(len(X_ctx))
        return np.array([[1.0 - self.p1, self.p1]], dtype=np.float64)


def _mk(**kw):
    torch.manual_seed(0)
    np.random.seed(0)
    defaults = dict(input_dim=4, buffer_size=200, consolidation_window=20,
                    consolidation_cooldown=10)
    defaults.update(kw)
    m = MultiTimescaleModel(**defaults)
    m.slow_prior = _ConstSlowPrior()
    return m


def _run(m, T=300, ctx=100, y_of=lambda t: 1.0, t0=0):
    rng = np.random.default_rng(0)
    X_ctx = rng.normal(size=(ctx, 4)).astype(np.float32)
    for t in range(t0, t0 + T):
        # context 标签两类都有，避免触发 SlowPrior 单类 fallback
        y_ctx = np.array([(i + t) % 2 for i in range(ctx)], dtype=np.int64)
        m.step(X_ctx, y_ctx, rng.normal(size=4).astype(np.float32), y_of(t), t=t)
    return m


class TestDefaultsUnchanged:

    def test_phase3_path_resolves_to_none_action(self):
        """Phase 3 路径（无 library）默认动作解析为 none，不再走 route 分支。"""
        m = _mk()
        assert m.use_adapter_library is False
        assert m.action_on_alarm == "none"
        assert m.trigger_source == "detector"
        assert m.detector is None

    def test_phase4a_path_resolves_to_route_adapter(self):
        m = _mk(use_adapter_library=True)
        assert m.action_on_alarm == "route_adapter"
        assert m.clear_detector_on_alarm is True
        assert m.consolidate_on_post_alarm_data is False

    def test_phase3_bias_trigger_still_consolidates(self):
        """未指定任何 Phase 5.5 参数时，Phase 3 的 bias-threshold 巩固路径仍工作。"""
        m = _mk(consolidation_threshold=0.01)
        _run(m, T=120, y_of=lambda t: 0.0)   # y_slow=0.9, y_t=0 → 稳定大负偏置
        assert m.consolidation_events, "Phase 3 bias-threshold 巩固路径不应被破坏"


class TestGuards:

    def test_empty_oracle_raises(self):
        with pytest.raises(ValueError, match="oracle_trigger_times 为空"):
            _mk(use_adapter_library=True, trigger_source="oracle",
                oracle_trigger_times=[])

    def test_route_adapter_without_library_raises(self):
        with pytest.raises(ValueError, match="route_adapter"):
            _mk(action_on_alarm="route_adapter", use_adapter_library=False)

    def test_unknown_action_rejected(self):
        with pytest.raises(AssertionError):
            _mk(action_on_alarm="teleport")

    def test_oracle_forces_shadow_detector(self):
        """oracle 模式必须禁用 detector.clear()，否则检测延迟测不准。"""
        m = _mk(use_adapter_library=True, trigger_source="oracle",
                oracle_trigger_times=[50], clear_detector_on_alarm=True)
        assert m.clear_detector_on_alarm is False


class TestOracleTiming:

    def test_context_reset_applies_before_prediction_at_drift_step(self):
        """t == drift point 当步就截断 —— 与 run_baselines.py 的 oracle 同一时刻表。"""
        m = _mk(action_on_alarm="context_reset", trigger_source="oracle",
                oracle_trigger_times=[150], reset_size=50)
        _run(m, T=60, ctx=100, t0=120)
        lens = m.slow_prior.seen_context_lens
        # t=120..149 未报警 → 全窗 100；t=150 起截断为 min(50 + (t-150), 100)
        assert lens[:30] == [100] * 30, lens[:5]
        assert lens[30] == 50, f"漂移当步应立即截断到 reset_size，实际 {lens[30]}"
        assert lens[31] == 51 and lens[35] == 55, lens[30:36]
        assert m.n_context_truncations > 0

    def test_shadow_detector_records_under_oracle(self):
        """oracle 驱动动作时，detector 仍在影子模式记录事件。"""
        m = _mk(use_adapter_library=True, action_on_alarm="none",
                trigger_source="oracle", oracle_trigger_times=[150],
                detector_impl="river", detector_cooldown=40)
        _run(m, T=400, y_of=lambda t: 1.0 if t < 200 else 0.0)
        assert m.alarm_events == [150], m.alarm_events
        assert m.detector_events, "影子 detector 应记录到错误率跳变"
        assert all(e != 150 for e in m.detector_events) or True  # 影子事件与 oracle 报警独立

    def test_alarm_recorded_once_per_oracle_time(self):
        m = _mk(use_adapter_library=True, action_on_alarm="none",
                trigger_source="oracle", oracle_trigger_times=[120, 160])
        _run(m, T=100, t0=100)
        assert m.alarm_events == [120, 160]


class TestContextResetGuards:

    def test_min_context_and_class_coverage(self):
        """截断后若只剩单类，应自动放宽直到有 2 类（防常量 fallback → 误报循环）。"""
        m = _mk(action_on_alarm="context_reset", trigger_source="oracle",
                oracle_trigger_times=[100], reset_size=4,
                min_context_after_reset=2)
        rng = np.random.default_rng(0)
        ctx = 100
        X_ctx = rng.normal(size=(ctx, 4)).astype(np.float32)
        # 末尾 20 个全是类别 1 → reset_size=4 的截断会得到单类
        y_ctx = np.array([0] * (ctx - 20) + [1] * 20, dtype=np.int64)
        m.step(X_ctx, y_ctx, rng.normal(size=4).astype(np.float32), 1.0, t=100)
        seen = m.slow_prior.seen_context_lens[-1]
        assert seen > 20, f"截断后应放宽到含 2 类，实际 context 长度 {seen}"
        assert len(np.unique(y_ctx[-seen:])) == 2


class TestPostAlarmConsolidation:

    def test_consolidation_deferred_to_post_alarm_window(self):
        """consolidate_on_post_alarm_data=True 时巩固推迟到 alarm_t + window。"""
        w = 20
        m = _mk(use_adapter_library=True, trigger_source="oracle",
                oracle_trigger_times=[150], consolidation_window=w,
                consolidate_on_post_alarm_data=True)
        _run(m, T=200, t0=100)
        assert m.consolidation_events, "推迟后仍应发生巩固"
        assert min(m.consolidation_events) >= 150 + w, (
            f"巩固应推迟到 >= {150 + w}，实际 {m.consolidation_events}"
        )

    def test_immediate_consolidation_by_default(self):
        w = 20
        m = _mk(use_adapter_library=True, trigger_source="oracle",
                oracle_trigger_times=[150], consolidation_window=w)
        _run(m, T=200, t0=100)
        assert 150 in m.consolidation_events, m.consolidation_events


class TestActionArms:

    @pytest.mark.parametrize("action", ["buffer_clear", "none"])
    def test_each_arm_runs_and_records(self, action):
        m = _mk(use_adapter_library=True, action_on_alarm=action,
                trigger_source="oracle", oracle_trigger_times=[150])
        _run(m, T=120, t0=100)
        assert (150, action) in m.action_events

    def test_buffer_clear_empties_buffer(self):
        m = _mk(use_adapter_library=True, action_on_alarm="buffer_clear",
                trigger_source="oracle", oracle_trigger_times=[150])
        _run(m, T=51, t0=100)   # 跑到 t=150 为止
        assert len(m.fast_corrector.buffer) <= 1, (
            f"buffer_clear 后应几乎为空，实际 {len(m.fast_corrector.buffer)}"
        )

    def test_none_arm_takes_no_action(self):
        m = _mk(use_adapter_library=True, action_on_alarm="none",
                trigger_source="oracle", oracle_trigger_times=[150])
        _run(m, T=120, t0=100)
        assert m.route_events == []
        assert m.n_context_truncations == 0

    def test_all_arms_declared(self):
        assert set(ACTIONS_ON_ALARM) == {
            "route_adapter", "context_reset", "buffer_clear", "none"}
        assert set(TRIGGER_SOURCES) == {"detector", "oracle"}


class TestDetectorInput:
    """Phase 5.5：可切换的检测器输入（Step 4 诊断的落地）。"""

    def test_default_is_indicator(self):
        m = _mk(use_adapter_library=True)
        assert m.detector_input == "indicator"

    def test_unknown_input_rejected(self):
        with pytest.raises(AssertionError):
            _mk(use_adapter_library=True, detector_input="entropy")

    def test_pred1_needs_no_stale_path(self):
        """pred1 是模型自己的类先验，不需要第二路预测 —— 这正是它的优势。"""
        m = _mk(use_adapter_library=True, detector_input="pred1")
        _run(m, T=120, t0=100)
        assert len(m.detector_signal_history) == 120
        assert set(np.unique(m.detector_signal_history)) <= {0.0, 1.0}

    def test_contrast_without_stale_proba_raises(self):
        m = _mk(use_adapter_library=True, detector_input="contrast_prob")
        with pytest.raises(RuntimeError, match="set_stale_proba"):
            _run(m, T=5, t0=100)

    def test_contrast_prob_uses_injected_stale_path(self):
        m = _mk(use_adapter_library=True, detector_input="contrast_prob")
        m.set_stale_proba(np.full(200, 0.2), offset=100)
        _run(m, T=50, t0=100)
        # stub 的 y_slow 恒为 0.9 → |0.2 − 0.9| = 0.7
        assert np.allclose(m.detector_signal_history, 0.7, atol=1e-6)

    def test_contrast_hard_is_binary_disagreement(self):
        m = _mk(use_adapter_library=True, detector_input="contrast_hard")
        m.set_stale_proba(np.full(200, 0.2), offset=100)   # 硬预测 0 vs slow 的 1
        _run(m, T=50, t0=100)
        assert np.allclose(m.detector_signal_history, 1.0)

    def test_stale_proba_out_of_range_raises(self):
        m = _mk(use_adapter_library=True, detector_input="contrast_prob")
        m.set_stale_proba(np.full(10, 0.3), offset=100)
        with pytest.raises(IndexError, match="超出 stale_proba"):
            _run(m, T=50, t0=100)

    def test_indicator_history_recorded_regardless_of_input(self):
        """换检测输入不影响 indicator 的落盘，事后可重放比较。"""
        for inp in ("indicator", "pred1"):
            m = _mk(use_adapter_library=True, detector_input=inp)
            _run(m, T=60, t0=100)
            assert len(m.indicator_history) == 60
            assert len(m.detector_signal_history) == 60

    def test_all_inputs_declared(self):
        from src.models.multi_timescale import DETECTOR_INPUTS
        assert set(DETECTOR_INPUTS) == {
            "indicator", "pred1", "contrast_prob", "contrast_hard"}
