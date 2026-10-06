"""
Phase 4 A 集成测试：MultiTimescaleModel(use_adapter_library=True)

不依赖 TabPFN —— 用 monkey-patched SlowPrior 注入受控的 y_slow 序列，
专注验证 detector + AdapterLibrary 的集成正确性：
  1. use_adapter_library=False 时 self.detector / self.adapter_library 为 None，
     行为完全等价 Phase 3 v2+B+F
  2. use_adapter_library=True 时 GatedEnsemble.adapter 已被替换为 AdapterLibrary
  3. 漂移信号注入后 detector_events / route_events 非空
  4. consolidation_events 在 detector 触发时同步增长（路由后立即巩固）
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from src.models.multi_timescale import MultiTimescaleModel
from src.regime.adapter_library import AdapterLibrary
from src.drift.error_detector import ADWINErrorDetector


class _StubSlowPrior:
    """
    Stand-in for SlowPrior：根据 _t 步阶段返回不同 y_slow，
    模拟 "regime 切换后 TabPFN 在新 regime 上初期错很多" 的现象。
    """

    def __init__(self):
        self._t = 0
        self.shift_at = 200
        self.rng = np.random.default_rng(0)

    def predict_proba(self, X_ctx, y_ctx, X_query):
        self._t += 1
        # 前 shift_at 步：y_slow 接近真实标签（错误率低）
        # shift_at 之后：y_slow 与真实标签反向（错误率高，制造 raw error 大幅偏离）
        # 这里我们只返回受控的概率；调用方的 y_t 决定 raw error
        if self._t < self.shift_at:
            p1 = 0.85   # 低错误率阶段
        else:
            p1 = 0.15   # 反向阶段
        return np.array([[1.0 - p1, p1]], dtype=np.float64)


class TestPhase4AIntegration:

    def test_default_path_unchanged(self):
        """use_adapter_library=False（默认）下行为与 Phase 3 v2+B+F 一致：
        detector / adapter_library 为 None，adapter_optimizer 不为 None。"""
        m = MultiTimescaleModel(input_dim=4)
        assert m.use_adapter_library is False
        assert m.detector is None
        assert m.adapter_library is None
        assert m.adapter_optimizer is not None
        # GatedEnsemble.adapter 仍是普通 Sequential
        assert not isinstance(m.gated_ensemble.adapter, AdapterLibrary)

    def test_phase4a_path_replaces_adapter(self):
        """use_adapter_library=True 下 detector + library 实例化、adapter 已替换。"""
        m = MultiTimescaleModel(
            input_dim=4,
            use_adapter_library=True,
            max_adapters=4,
            library_fit_threshold=0.05,
        )
        assert isinstance(m.detector, ADWINErrorDetector)
        assert isinstance(m.adapter_library, AdapterLibrary)
        assert m.adapter_optimizer is None
        # drop-in 替换
        assert m.gated_ensemble.adapter is m.adapter_library

    def test_drift_triggers_route_and_consolidate(self):
        """注入受控漂移信号，detector_events / route_events / consolidation_events 都应非空。"""
        torch.manual_seed(0)
        np.random.seed(0)

        m = MultiTimescaleModel(
            input_dim=4,
            buffer_size=200,
            use_adapter_library=True,
            max_adapters=4,
            library_fit_threshold=0.05,
            consolidation_window=50,
            consolidation_threshold=0.05,
            consolidation_cooldown=80,
            detector_min_subwindow=30,
            detector_cooldown=60,
        )
        # monkey-patch SlowPrior，避免 TabPFN
        m.slow_prior = _StubSlowPrior()

        rng = np.random.default_rng(7)
        X_ctx_dummy = rng.normal(size=(20, 4)).astype(np.float32)
        y_ctx_dummy = rng.integers(0, 2, size=20).astype(np.int64)

        T = 400
        for t in range(T):
            x_t = rng.normal(size=4).astype(np.float32)
            # 真实标签：前 shift_at 步随机；shift_at 后偏向 1（与 stub 反向制造 large error）
            if t < 200:
                y_t = float(rng.integers(0, 2))
            else:
                y_t = 1.0  # 之后 stub 给 0.15，raw error ≈ +0.85
            m.step(X_ctx_dummy, y_ctx_dummy, x_t, y_t, t=t)

        assert m.detector.t == T, f"detector 应见过 {T} 步，实见 {m.detector.t}"
        assert len(m.detector_events) >= 1, (
            f"应至少检测到 1 次漂移，实际 {len(m.detector_events)}: {m.detector_events}"
        )
        assert len(m.route_events) >= 1, (
            f"漂移应触发至少 1 次 route，实际 {len(m.route_events)}"
        )
        assert len(m.consolidation_events) >= 1, (
            f"漂移触发后应至少巩固 1 次，实际 {len(m.consolidation_events)}"
        )

        # 第一次 detector 触发的 t 应在 shift_at=200 之后（在新 regime 内累积充足后）
        first_detect = m.detector_events[0]
        assert first_detect >= 200, (
            f"首次漂移应在 t≥200 检测到，实际 t={first_detect}"
        )

        # route 后 library 至少应保留 1 个 adapter；可能新建（→ ≥2）
        assert m.adapter_library.n_adapters() >= 1
