"""Round 2 运行器：批量 prequential + 留出集回测（协议见 results/phase56_prereg.md）。

一次运行 = 一种记忆方法 × 一个 seed。流程：
  1. 从流内第 `start` 行起，每 `stride` 行一批（批不跨阶段边界）：
     用"此前已揭示的行"构造上下文 → 预测这一批 → 揭示标签、更新记忆。
  2. 阶段在官方变点（或 ADWIN 报警）处关闭，档案类方法此时存档。
  3. 每 `ckpt_every` 行和每个官方阶段结束时，用**当前**记忆预测所有已出现阶段的留出集。
     回测不改变记忆（前后 digest 必须相同）。

不变量（违反即抛错，不会产出看似正常的结果）：
  - 上下文里每一行的流内下标都 < 当前批的第一行（不偷看未来、不含被预测行）；
  - 上下文大小不超过该方法的预算；
  - 回测前后记忆的 digest 不变。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from src.memory.context_memory import NO_ROUTE, RegimeArchive


@dataclass
class RunResult:
    pred: np.ndarray                 # (n_stream,) int8，未预测的行为 -1
    route: np.ndarray                # (n_stream,) int8，NO_ROUTE / WINDOW(-1) / 档案编号
    ckpt_t: np.ndarray               # (C,) 回测时刻（流内下标，= 已揭示行数）
    ckpt_upto: np.ndarray            # (C,) 回测时已出现的最大官方阶段
    ckpt_pred: np.ndarray            # (C, H) int8，-1 = 该留出行在此检查点未评估
    closures: np.ndarray             # 阶段关闭的流内位置（关闭发生在该行之前）
    n_fits: int = 0
    fit_seconds: float = 0.0
    max_ctx: int = 0
    leak_count: int = 0
    digests: "list[str]" = field(default_factory=list)


def run_stream(
    memory,
    predict_fn,
    X: np.ndarray,
    y: np.ndarray,
    regime: np.ndarray,
    X_hold: np.ndarray,
    regime_hold: np.ndarray,
    boundaries: "list[int]",
    *,
    stride: int = 50,
    start: int = 200,
    ckpt_every: int = 500,
    segmentation: str = "official",
    detector=None,
    min_segment: int = 500,
    max_rows: "int | None" = None,
    progress=None,
) -> RunResult:
    """boundaries：官方变点在流内坐标下的位置（新阶段第一行）。"""
    n = len(y) if max_rows is None else min(len(y), int(max_rows))
    budget = memory.max_context()
    pred = np.full(len(y), -1, dtype=np.int8)
    route = np.full(len(y), NO_ROUTE, dtype=np.int8)
    ckpt_t, ckpt_upto, ckpt_rows, closures, digests = [], [], [], [], []
    res = RunResult(pred=pred, route=route, ckpt_t=None, ckpt_upto=None, ckpt_pred=None,
                    closures=None)
    bnds = sorted(int(b) for b in boundaries if 0 < b < n)
    if segmentation not in ("official", "adwin"):
        raise ValueError(segmentation)
    if segmentation == "adwin" and detector is None:
        raise ValueError("segmentation='adwin' 需要 detector")

    def predict_groups(Xq, t_limit):
        out = np.empty(len(Xq), dtype=np.int64)
        rts = np.full(len(Xq), NO_ROUTE, dtype=np.int64)
        for pos, ctx, r in memory.contexts(Xq):
            if len(ctx) == 0:
                raise RuntimeError("上下文为空")
            if len(ctx) > budget:
                raise RuntimeError(f"上下文 {len(ctx)} 行超出预算 {budget}")
            if ctx.max() >= t_limit:
                res.leak_count += int((ctx >= t_limit).sum())
                raise RuntimeError(f"上下文含未揭示的行（max {ctx.max()} ≥ {t_limit}）")
            res.max_ctx = max(res.max_ctx, len(ctx))
            t0 = time.time()
            proba = predict_fn(X[ctx], y[ctx], Xq[pos])
            res.fit_seconds += time.time() - t0
            res.n_fits += 1
            out[pos] = np.argmax(proba, axis=1)
            rts[pos] = r
        return out, rts

    def checkpoint(t):
        upto = int(regime[t - 1])
        sel = np.flatnonzero(regime_hold <= upto)
        before = memory.digest()
        p, _ = predict_groups(X_hold[sel], t)
        if memory.digest() != before:
            raise RuntimeError("回测改变了记忆状态")
        row = np.full(len(X_hold), -1, dtype=np.int8)
        row[sel] = p
        ckpt_t.append(t); ckpt_upto.append(upto); ckpt_rows.append(row); digests.append(before)

    memory.update(np.arange(0, start))
    seg_start = 0
    next_ckpt = (start // ckpt_every + 1) * ckpt_every
    t = start
    while t < n:
        nxt = [b for b in bnds if b > t]
        end = min(t + stride, n, nxt[0] if nxt else n)
        q = np.arange(t, end)
        p, rts = predict_groups(X[q], t)
        pred[q] = p
        route[q] = rts
        memory.update(q)

        if segmentation == "adwin":
            err = (p != y[q]).astype(float)
            for j, e in enumerate(err):
                row_t = t + j
                if detector.update(float(e)) and row_t + 1 - seg_start >= min_segment:
                    memory.close_regime(np.arange(seg_start, row_t + 1))
                    closures.append(row_t + 1)
                    seg_start = row_t + 1

        regime_end = end in bnds or end == n
        if segmentation == "official" and end in bnds:
            memory.close_regime(np.arange(seg_start, end))
            closures.append(end)
            seg_start = end
        if end >= next_ckpt or regime_end:
            checkpoint(end)
            while next_ckpt <= end:
                next_ckpt += ckpt_every
        if progress is not None:
            progress(end, n, res)
        t = end

    res.ckpt_t = np.asarray(ckpt_t, dtype=np.int64)
    res.ckpt_upto = np.asarray(ckpt_upto, dtype=np.int64)
    res.ckpt_pred = np.stack(ckpt_rows) if ckpt_rows else np.zeros((0, len(X_hold)), np.int8)
    res.closures = np.asarray(closures, dtype=np.int64)
    res.digests = digests
    return res


def needs_detector(method: str) -> bool:
    return method.endswith("_adwin")


def is_archive(memory) -> bool:
    return isinstance(memory, RegimeArchive)
