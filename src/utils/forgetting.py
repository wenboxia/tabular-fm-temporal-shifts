"""
适应 / 遗忘 trade-off 的回测工具（Phase 5.5 Step 6，维度 C）

目标：在适应能力相近的前提下，看能否减少对早期概念的遗忘。

即在流的若干检查点，用**当前模型**回测更早概念的留出样本，看它还记不记得。

三条硬要求（错一条整个指标就没意义）：

1. **留出样本不能进主循环**。回测集必须从 prequential 流里剔除，
   否则它既进过 TabPFN 的 context、又进过 buffer、还参与过 gate 训练，
   那测的是拟合而不是遗忘。`carve_holdout()` 负责切分并返回剩余流。
2. **回测不能污染模型状态**。回测要跑前向，但绝不能更新 gate / adapter / buffer /
   detector 的任何状态。`backtest()` 用 `frozen()` 上下文管理器保证这点，
   并由 `state_hash()` 在回测前后比对做断言。
3. **回测要用当时的 context**。TabPFN 是 in-context learner，"记不记得旧概念"
   取决于喂给它什么 context。回测时用**回测集自己的**前若干样本作 context，
   这样测的是"当前的 gate/adapter 参数在旧概念上还好不好使"，
   而不是"当前 context 窗口里恰好有没有旧概念样本"。
"""

from __future__ import annotations

import contextlib
import hashlib
from dataclasses import dataclass, field

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 留出集切分
# ---------------------------------------------------------------------------


@dataclass
class Holdout:
    """一个概念的留出回测集。"""

    name: str
    X: np.ndarray
    y: np.ndarray
    origin_t: int              # 该概念在原始流里的起始位置（仅作标注）
    context_size: int = 100    # 回测时取自身前多少样本作 TabPFN context

    def __post_init__(self):
        if len(self.X) != len(self.y):
            raise ValueError(f"X/y 长度不一致: {len(self.X)} vs {len(self.y)}")
        if len(self.X) <= self.context_size:
            raise ValueError(
                f"holdout {self.name!r} 只有 {len(self.X)} 个样本，"
                f"不足以留出 context_size={self.context_size} 后再评估"
            )
        if len(np.unique(self.y[: self.context_size])) < 2:
            raise ValueError(
                f"holdout {self.name!r} 的 context 部分只有单一类别，"
                "TabPFN 会退化成常量预测，回测无意义"
            )

    def __len__(self) -> int:
        return len(self.X) - self.context_size


def carve_holdout(
    X: np.ndarray,
    y: np.ndarray,
    regions: "list[tuple[str, int, int]]",
    holdout_size: int = 300,
    context_size: int = 100,
    seed: int = 42,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, list[Holdout]]":
    """从流中切出若干概念的留出集，并返回**剔除它们之后**的流。

    Args:
        X, y:         完整段
        regions:      [(概念名, lo, hi), ...]，半开区间，互不重叠
        holdout_size: 每个概念留出多少样本
        context_size: 回测时每个留出集自留多少样本作 context
        seed:         留出样本的采样种子

    Returns:
        X_stream, y_stream: 剔除留出样本后的流（**主循环用这个**）
        kept_idx:           保留下来的原始下标（用于把漂移点重映射到新坐标）
        holdouts:           每个概念一个 Holdout

    留出样本在概念区间内**连续取自区间末尾**，而不是随机散点：
    随机散点会在流里留下密集空洞，把相邻样本的时间间隔改变；
    取末尾一段则只是把该概念截短，时间结构不变。
    """
    if len(X) != len(y):
        raise ValueError(f"X/y 长度不一致: {len(X)} vs {len(y)}")
    _ = np.random.default_rng(seed)   # 保留接口，当前策略是确定性的

    spans = sorted(regions, key=lambda r: r[1])
    for (n0, _, h0), (n1, l1, _) in zip(spans[:-1], spans[1:]):
        if h0 > l1:
            raise ValueError(f"概念区间重叠: {n0} 与 {n1}")

    drop = np.zeros(len(X), dtype=bool)
    holdouts: list[Holdout] = []
    for name, lo, hi in spans:
        if not (0 <= lo < hi <= len(X)):
            raise ValueError(f"概念 {name!r} 的区间 [{lo},{hi}) 越界（n={len(X)}）")
        if hi - lo < holdout_size:
            raise ValueError(
                f"概念 {name!r} 只有 {hi - lo} 个样本，不够留出 {holdout_size} 个"
            )
        h_lo = hi - holdout_size
        drop[h_lo:hi] = True
        holdouts.append(Holdout(
            name=name, X=X[h_lo:hi].copy(), y=y[h_lo:hi].copy(),
            origin_t=h_lo, context_size=context_size,
        ))

    kept_idx = np.flatnonzero(~drop)
    return X[kept_idx].copy(), y[kept_idx].copy(), kept_idx, holdouts


def carve_regime_holdouts(
    y: np.ndarray,
    regime: np.ndarray,
    per_class: "dict[int, int] | int",
    seed: int,
) -> "tuple[np.ndarray, np.ndarray]":
    """Round 2：每个阶段按类别分层、在**整个阶段**里均匀随机抽留出集。

    与 `carve_holdout`（取阶段末尾一段）不同：Insects 每个阶段的末尾恰好是一段单类长块，
    取末尾会得到几乎全是同一类的留出集。这里在整个阶段内按类随机抽，留出集的类别分布
    与阶段整体一致（abrupt_balanced 每个阶段各类数量相等）。

    Args:
        y:          全流标签（原始顺序）
        regime:     每行所属阶段
        per_class:  每阶段每类抽多少；int = 所有阶段相同，dict = {阶段: 每类数量}
        seed:       抽样种子

    Returns:
        stream_idx:  保留在流里的原始下标（升序）
        holdout_idx: 留出的原始下标（按阶段、再按下标升序）
    """
    y = np.asarray(y); regime = np.asarray(regime)
    if len(y) != len(regime):
        raise ValueError("y 与 regime 长度不一致")
    rng = np.random.default_rng(seed)
    picked = []
    for r in np.unique(regime):
        k = per_class[int(r)] if isinstance(per_class, dict) else int(per_class)
        rows_r = np.flatnonzero(regime == r)
        for c in np.unique(y[rows_r]):
            rows_rc = rows_r[y[rows_r] == c]
            if len(rows_rc) < k:
                raise ValueError(f"阶段 {r} 类别 {c} 只有 {len(rows_rc)} 行，不够留出 {k}")
            picked.append(np.sort(rng.choice(rows_rc, size=k, replace=False)))
    holdout_idx = np.sort(np.concatenate(picked)) if picked else np.zeros(0, np.int64)
    holdout_idx = holdout_idx[np.lexsort((holdout_idx, regime[holdout_idx]))]
    mask = np.ones(len(y), dtype=bool); mask[holdout_idx] = False
    return np.flatnonzero(mask), holdout_idx.astype(np.int64)


def remap_points(points: "list[int]", kept_idx: np.ndarray) -> "list[int]":
    """把原坐标下的时刻（如漂移点）映射到剔除留出样本之后的新坐标。"""
    return [int(np.searchsorted(kept_idx, p)) for p in points]


# ---------------------------------------------------------------------------
# 状态哈希 + 冻结
# ---------------------------------------------------------------------------


def state_hash(model) -> str:
    """对模型的**全部可变状态**取哈希，用于断言回测没有污染主循环。

    覆盖：gate / adapter（或 adapter library 全部 adapter）参数、
    optimizer 的 step 计数、buffer 内容、detector 内部计数、各事件列表长度。
    """
    h = hashlib.sha256()

    def _upd(*vals):
        for v in vals:
            h.update(repr(v).encode())

    for name, p in sorted(model.gated_ensemble.named_parameters()):
        _upd(name, np.asarray(p.detach().numpy()).tobytes())

    for opt in (getattr(model, "gate_optimizer", None),
                getattr(model, "adapter_optimizer", None)):
        if opt is not None:
            _upd([len(opt.state)] + [
                int(s.get("step", 0)) if isinstance(s.get("step", 0), int)
                else float(s.get("step", 0))
                for s in opt.state.values()
            ])

    buf = model.fast_corrector.buffer
    _upd(len(buf), list(np.round(np.asarray(buf.recent_errors(len(buf))), 8))
         if len(buf) else [])

    det = getattr(model, "detector", None)
    if det is not None:
        _upd(det.t, det.n_drifts, len(det), round(det.current_mean(), 8))

    lib = getattr(model, "adapter_library", None)
    if lib is not None:
        _upd(lib.n_adapters(), lib.active_id)

    _upd(len(model.consolidation_events), len(model.detector_events),
         len(model.route_events), len(getattr(model, "alarm_events", [])),
         len(model.indicator_history), len(model.abs_error_history),
         model._last_consolidation_t, getattr(model, "_last_alarm_t", None))
    return h.hexdigest()


@contextlib.contextmanager
def frozen(model):
    """回测期间冻结模型：no_grad + eval，并在退出时断言状态未变。"""
    before = state_hash(model)
    was_training = model.gated_ensemble.training
    model.gated_ensemble.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        if was_training:
            model.gated_ensemble.train()
        after = state_hash(model)
        if before != after:
            raise RuntimeError(
                "回测污染了模型状态（state_hash 变了）。"
                "回测只能做前向，绝不能更新 gate / adapter / buffer / detector。"
            )


# ---------------------------------------------------------------------------
# 回测
# ---------------------------------------------------------------------------


@dataclass
class BacktestPoint:
    """一次检查点上、一个留出集的回测结果。"""

    t: int
    holdout: str
    accuracy: float
    n: int


@dataclass
class ForgettingTracker:
    """在若干检查点上回测所有留出集，产出遗忘曲线。"""

    holdouts: "list[Holdout]"
    every: int = 500
    points: "list[BacktestPoint]" = field(default_factory=list)
    _baseline: dict = field(default_factory=dict)

    def maybe_backtest(self, model, t: int, force: bool = False) -> bool:
        """到检查点就回测一次；返回本次是否真的跑了。"""
        if not force and (t % self.every != 0):
            return False
        for h in self.holdouts:
            acc = backtest_accuracy(model, h)
            self.points.append(BacktestPoint(t=t, holdout=h.name, accuracy=acc, n=len(h)))
            self._baseline.setdefault(h.name, acc)
        return True

    def curve(self, holdout_name: str) -> "tuple[list[int], list[float]]":
        pts = [p for p in self.points if p.holdout == holdout_name]
        return [p.t for p in pts], [p.accuracy for p in pts]

    def forgetting(self, holdout_name: str) -> "float | None":
        """遗忘量 = 该留出集上的**最高**历史准确率 − 最终准确率。

        用最高值而不是第一次的值作基准，是 continual learning 的通行定义
        （backward transfer）：模型可能先变好再变差，遗忘要从它的最好状态起算。
        """
        _, accs = self.curve(holdout_name)
        if len(accs) < 2:
            return None
        return float(max(accs) - accs[-1])

    def summary(self) -> dict:
        return {
            h.name: {
                "first": self.curve(h.name)[1][0] if self.curve(h.name)[1] else None,
                "best": max(self.curve(h.name)[1]) if self.curve(h.name)[1] else None,
                "final": self.curve(h.name)[1][-1] if self.curve(h.name)[1] else None,
                "forgetting": self.forgetting(h.name),
                "n": len(h),
            }
            for h in self.holdouts
        }


def backtest_accuracy(model, holdout: Holdout) -> float:
    """用当前模型在一个留出集上评估准确率，全程零副作用。

    context 取该留出集**自己的**前 context_size 个样本，评估其余样本。
    这样测的是"当前参数在旧概念上还好不好使"，而不是主循环的 context 里
    恰好还剩多少旧概念样本。
    """
    X_ctx = holdout.X[: holdout.context_size]
    y_ctx = holdout.y[: holdout.context_size]
    X_q = holdout.X[holdout.context_size:]
    y_q = holdout.y[holdout.context_size:]

    with frozen(model):
        proba = model.slow_prior.predict_proba(X_ctx, y_ctx, X_q)
        y_slow = torch.tensor(proba[:, 1:2], dtype=torch.float32)      # (n, 1)
        x = torch.tensor(np.asarray(X_q, dtype=np.float32))            # (n, D)
        # 回测不更新 buffer，所以 correction 用零：buffer 里装的是**当前**概念的
        # 残差，把它施加到旧概念样本上是张冠李戴，会污染遗忘的测量。
        correction = torch.zeros_like(y_slow)
        y_raw, _ = model.gated_ensemble(x, y_slow, correction)
        preds = (torch.clamp(y_raw, 0.0, 1.0).squeeze(-1).numpy() >= 0.5).astype(int)

    return float(np.mean(preds == y_q))
