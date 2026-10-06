"""
Phase 4 A — ADWIN-style 漂移检测器（1D 误差流）

在 1D 标量误差流（如 y_t - y_slow ∈ [-1, 1]）上运行 ADWIN 风格变点检测。
不依赖 river / scikit-multiflow，自包含实现。

算法概要（Bifet & Gavaldà 2007 简化版）：
  1. 维护一个 FIFO 窗口 W，每步压入新观测
  2. 当 |W| ≥ 2 · min_subwindow 后，遍历可能的切点 i，
     将 W 划分为 W0 = W[:i] 与 W1 = W[i:]
  3. 若存在某 i 使 |mean(W0) - mean(W1)| > ε_cut，
     则声明漂移，丢弃 W0（旧部分），保留 W1
  4. ε_cut 由 Hoeffding 界给出（误差 ∈ [-1, 1] → range = 2）：
        ε_cut = range · sqrt(0.5 · (1/n0 + 1/n1) · ln(2/δ'))
     δ' = δ / max(1, log(n))（多重比较 Bonferroni-like 校正）

使用：
    detector = ADWINErrorDetector(delta=0.002, min_subwindow=30)
    for t in range(T):
        drift = detector.update(error_t)   # bool
        if drift:
            print(f"drift at t={t}")

设计取舍：
  - 切点扫描 O(n)/步，对 T ≤ 5000、窗口 ≤ 1000 完全够（CPU 数百毫秒级）
  - cooldown：声明漂移后强制冷却若干步，防止同一漂移被反复触发
  - clear()：外部可在 routing 后清空，强制 detector 从新 regime 重新积累
"""

import math
from collections import deque
from typing import Optional

import numpy as np


class ADWINErrorDetector:
    """
    1D 标量误差流上的 ADWIN 风格漂移检测器。

    输入误差应为有界标量（默认假设 |x| ≤ 1，对应 y_t - y_slow ∈ [-1, 1]）。
    每次 update(x) 返回 bool 指示本步是否发生漂移。
    """

    def __init__(
        self,
        delta: float = 0.002,
        min_subwindow: int = 30,
        max_window: int = 1000,
        value_range: float = 2.0,
        cooldown: int = 50,
    ):
        """
        Args:
            delta:         置信参数（越小越保守，漂移触发越难）
            min_subwindow: 切点两侧子窗口最小样本数（防早期噪声误触发）
            max_window:    主窗口上限（超过则丢弃最旧）
            value_range:   误差取值范围 b - a，用于 Hoeffding 界。
                           默认 2.0（对应 [-1, 1]）；若误差 ∈ [0, 1] 应传 1.0。
            cooldown:      漂移声明后强制冷却步数（期间不再检测）
        """
        assert 0.0 < delta < 1.0, f"delta 必须 ∈ (0,1)，收到: {delta}"
        assert min_subwindow >= 5, f"min_subwindow 太小: {min_subwindow}"
        assert max_window >= 2 * min_subwindow, (
            f"max_window={max_window} 必须 ≥ 2·min_subwindow={2*min_subwindow}"
        )
        assert value_range > 0, f"value_range 必须 > 0，收到: {value_range}"
        assert cooldown >= 0, f"cooldown 必须 ≥ 0，收到: {cooldown}"

        self.delta = delta
        self.min_subwindow = min_subwindow
        self.max_window = max_window
        self.value_range = value_range
        self.cooldown = cooldown

        self._window: deque = deque()
        self._t: int = 0                  # 全局观测计数
        self._n_drifts: int = 0
        self._last_drift_t: int = -10**9
        self._cooldown_until: int = -1    # 在该 t 之前不检测

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def update(self, x: float) -> bool:
        """
        压入一个新观测，返回是否检测到漂移。

        Args:
            x: 标量误差（建议 ∈ [-1, 1]）

        Returns:
            True 表示本步声明漂移；False 否则。
        """
        x = float(x)
        self._t += 1
        self._window.append(x)

        # 窗口长度上限
        if len(self._window) > self.max_window:
            self._window.popleft()

        # 冷却期：跳过检测但仍累积观测
        if self._t < self._cooldown_until:
            return False

        n = len(self._window)
        if n < 2 * self.min_subwindow:
            return False

        # 累积和（用于 O(1) 求子窗口均值）
        values = np.fromiter(self._window, dtype=np.float64, count=n)
        cum = np.empty(n + 1, dtype=np.float64)
        cum[0] = 0.0
        np.cumsum(values, out=cum[1:])

        # δ'：Bonferroni-like 多重比较校正
        delta_prime = self.delta / max(1.0, math.log(max(2, n)))
        ln_2_dp = math.log(2.0 / delta_prime)

        # 扫描切点 i：W0 = W[:i]，W1 = W[i:]
        cut_at: int = -1
        for i in range(self.min_subwindow, n - self.min_subwindow + 1):
            n0 = i
            n1 = n - i
            mu0 = cum[i] / n0
            mu1 = (cum[n] - cum[i]) / n1
            m_inv = 1.0 / n0 + 1.0 / n1
            epsilon = self.value_range * math.sqrt(0.5 * m_inv * ln_2_dp)
            if abs(mu0 - mu1) > epsilon:
                cut_at = i
                break

        if cut_at < 0:
            return False

        # 漂移：丢弃 W0，保留 W1
        for _ in range(cut_at):
            self._window.popleft()
        self._n_drifts += 1
        self._last_drift_t = self._t
        self._cooldown_until = self._t + self.cooldown
        return True

    def clear(self) -> None:
        """完全清空窗口（routing 后由调用方触发，迫使 detector 从零积累新 regime）。"""
        self._window.clear()
        self._cooldown_until = self._t + self.cooldown

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._window)

    @property
    def n_drifts(self) -> int:
        return self._n_drifts

    @property
    def last_drift_t(self) -> int:
        return self._last_drift_t

    @property
    def t(self) -> int:
        return self._t

    def current_mean(self) -> float:
        if not self._window:
            return 0.0
        return float(np.mean(self._window))

    def __repr__(self) -> str:
        return (
            f"ADWINErrorDetector("
            f"delta={self.delta}, min_sub={self.min_subwindow}, "
            f"window={len(self._window)}/{self.max_window}, "
            f"t={self._t}, n_drifts={self._n_drifts}, "
            f"last_drift_t={self._last_drift_t})"
        )


class RiverADWINDetector:
    """
    `river.drift.ADWIN` 的薄包装，对外接口与 ADWINErrorDetector 完全一致
    （update / clear / __len__ / n_drifts / last_drift_t / t / current_mean）。

    为什么需要它（Phase 5.5，2026-09-06）：
      自写版用 value_range·Hoeffding 界，不用经验方差；默认配置
      （max_window=400, min_subwindow=30, δ=0.002, value_range=1.0）下即使最有利的
      200/200 切分也要求 |Δmean| ≥ 0.209。真实 Insects 的 0/1 indicator 流错误率仅
      2–4%，官方变点 19500 处位移 ≈ 0.10，结构性不可能触发。
      river 用经验方差的 Bernstein 型界，对低错误率流阈值低得多（对比见
      tests/test_error_detector.py）。

    语义差异（解读 detector 消融时要注意）：
      - river 内部无 cooldown；本包装在 cooldown 期内**吞掉**报警（计入 n_suppressed），
        而自写版在 cooldown 期内跳过检测但窗口继续累积。
      - clear() 用 river 的 clone() 重建一个同参数的空检测器。
    """

    def __init__(
        self,
        delta: float = 0.002,
        cooldown: int = 50,
        clock: int = 1,
        min_window_length: int = 5,
        grace_period: int = 10,
        max_buckets: int = 5,
    ):
        try:
            from river import drift as _river_drift
        except ImportError as e:  # pragma: no cover
            raise ImportError("detector_impl='river' 需要 `pip install river`") from e
        assert 0.0 < delta < 1.0, f"delta 必须 ∈ (0,1)，收到: {delta}"
        assert cooldown >= 0, f"cooldown 必须 ≥ 0，收到: {cooldown}"
        assert clock >= 1, f"clock 必须 ≥ 1，收到: {clock}"
        self.delta = delta
        self.cooldown = cooldown
        self.clock = clock
        self._adwin = _river_drift.ADWIN(
            delta=delta,
            clock=clock,
            max_buckets=max_buckets,
            min_window_length=min_window_length,
            grace_period=grace_period,
        )
        self._t: int = 0
        self._n_drifts: int = 0
        self._n_suppressed: int = 0
        self._last_drift_t: int = -10**9
        self._cooldown_until: int = -1

    def update(self, x: float) -> bool:
        self._t += 1
        self._adwin.update(float(x))
        if not self._adwin.drift_detected:
            return False
        if self._t < self._cooldown_until:
            self._n_suppressed += 1
            return False
        self._n_drifts += 1
        self._last_drift_t = self._t
        self._cooldown_until = self._t + self.cooldown
        return True

    def clear(self) -> None:
        self._adwin = self._adwin.clone()
        self._cooldown_until = self._t + self.cooldown

    def __len__(self) -> int:
        return int(self._adwin.width)

    @property
    def n_drifts(self) -> int:
        return self._n_drifts

    @property
    def n_suppressed(self) -> int:
        return self._n_suppressed

    @property
    def last_drift_t(self) -> int:
        return self._last_drift_t

    @property
    def t(self) -> int:
        return self._t

    def current_mean(self) -> float:
        return float(self._adwin.estimation) if len(self) else 0.0

    def __repr__(self) -> str:
        return (
            f"RiverADWINDetector(delta={self.delta}, clock={self.clock}, "
            f"window={len(self)}, t={self._t}, n_drifts={self._n_drifts}, "
            f"n_suppressed={self._n_suppressed})"
        )


DETECTOR_IMPLS = ("own", "river")


def make_detector(
    impl: str = "own",
    *,
    delta: float = 0.002,
    min_subwindow: int = 30,
    max_window: int = 1000,
    value_range: float = 1.0,
    cooldown: int = 50,
    clock: int = 1,
):
    """按 impl 构造检测器。'own' = 自写 Hoeffding 版（Phase 4/5 既有行为）；'river' = 标准 ADWIN。"""
    if impl == "own":
        return ADWINErrorDetector(
            delta=delta, min_subwindow=min_subwindow, max_window=max_window,
            value_range=value_range, cooldown=cooldown,
        )
    if impl == "river":
        return RiverADWINDetector(delta=delta, cooldown=cooldown, clock=clock)
    raise ValueError(f"detector_impl 必须 ∈ {DETECTOR_IMPLS}，收到: {impl!r}")
