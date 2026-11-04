"""Round 2 指标（定义见 results/phase56_prereg.md）。

ADAPT  数据流上所有已预测行的平衡准确率（masked 版本剔除单类长段内的行）
RET    每个已结束阶段 j 的留出集平衡准确率，在 j 结束之后的所有检查点上取平均，再对 j 取平均
POST_d 变点 d 之后前 500 行的平衡准确率
FGT_j  留出集 j 的最佳准确率 − 最终准确率
RET_segL（第五轮，与阶段无关）把流按固定 L 行分段，段内留出行在该段结束 lag 行之后的检查点上的平均准确率，再对段等权平均
"""
from __future__ import annotations

import numpy as np

RUN_MASK_MIN = 30          # 长度 ≥ 30 的同类连续段视为"单类长段"
POST_ROWS = 500


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan")
    recalls = [float((y_pred[y_true == c] == c).mean()) for c in np.unique(y_true)]
    return float(np.mean(recalls))


def single_class_run_mask(y: np.ndarray, min_len: int = RUN_MASK_MIN) -> np.ndarray:
    """True = 该行位于长度 ≥ min_len 的同类连续段内。"""
    y = np.asarray(y)
    mask = np.zeros(len(y), dtype=bool)
    i = 0
    while i < len(y):
        j = i
        while j + 1 < len(y) and y[j + 1] == y[i]:
            j += 1
        if j - i + 1 >= min_len:
            mask[i: j + 1] = True
        i = j + 1
    return mask


def adapt(d, masked: bool = False) -> float:
    done = d["pred"] >= 0
    if masked:
        done &= ~single_class_run_mask(d["y"])
    return balanced_accuracy(d["y"][done], d["pred"][done])


def post_drift(d, boundary: int, masked: bool = False, rows: int = POST_ROWS) -> float:
    sl = np.zeros(len(d["y"]), dtype=bool)
    sl[boundary: boundary + rows] = True
    sl &= d["pred"] >= 0
    if masked:
        sl &= ~single_class_run_mask(d["y"])
    return balanced_accuracy(d["y"][sl], d["pred"][sl])


def holdout_curve(d, j: int) -> "tuple[np.ndarray, np.ndarray]":
    """阶段 j 留出集在各检查点的平衡准确率（只取评估过它的检查点）。"""
    sel = d["regime_hold"] == j
    ts, accs = [], []
    for c, t in enumerate(d["ckpt_t"]):
        row = d["ckpt_pred"][c, sel]
        if (row >= 0).all():
            ts.append(int(t)); accs.append(balanced_accuracy(d["y_hold"][sel], row))
    return np.asarray(ts), np.asarray(accs)


def regime_end(d, j: int) -> int:
    b = list(d["boundaries"])
    return int(b[j]) if j < len(b) else len(d["y"])


def ret_per_regime(d) -> "dict[int, float]":
    """阶段 j 结束之后（检查点时刻 > 结束位置）的平均留出准确率。"""
    out = {}
    for j in range(len(d["boundaries"])):
        ts, accs = holdout_curve(d, j)
        after = ts > regime_end(d, j)
        if after.any():
            out[j] = float(accs[after].mean())
    return out


def ret(d) -> float:
    r = ret_per_regime(d)
    return float(np.mean(list(r.values()))) if r else float("nan")


def holdout_positions(d) -> np.ndarray:
    """每个留出行在流坐标里的位置：它前面有多少个流行（stream_idx 与 hold_idx 都是原数据集里的下标，stream_idx 升序）。"""
    return np.searchsorted(np.asarray(d["stream_idx"]), np.asarray(d["hold_idx"]))


def ret_segments_per_segment(d, L: int = 2000, lag: int = 0) -> "dict[int, float]":
    """固定分段的保留（与阶段边界无关）。段 s = 流坐标 [L·s, L·(s+1))，只取完整段（L·(s+1) ≤ 流长）；
    段内留出行 = 位置落在该段里的留出行（没有留出行的段跳过）。
    在所有 ckpt_t > L·(s+1) + lag、且这些行都已预测（ckpt_pred ≥ 0）的检查点上算平衡准确率，取平均。
    返回 {s: 值}，只含有这样的检查点的段。"""
    assert L > 0 and lag >= 0
    pos = holdout_positions(d)
    n = len(d["y"])
    ts = np.asarray(d["ckpt_t"])
    out = {}
    for s in range(n // L):
        a, b = L * s, L * (s + 1)
        sel = (pos >= a) & (pos < b)
        if not sel.any():
            continue
        accs = []
        for c, t in enumerate(ts):
            if t <= b + lag:
                continue
            row = d["ckpt_pred"][c, sel]
            if (row >= 0).all():
                accs.append(balanced_accuracy(d["y_hold"][sel], row))
        if accs:
            out[s] = float(np.mean(accs))
    return out


def ret_segments(d, L: int = 2000, lag: int = 0) -> float:
    """ret_segments_per_segment 对段等权平均；没有可用段时为 nan。"""
    r = ret_segments_per_segment(d, L, lag)
    return float(np.mean(list(r.values()))) if r else float("nan")


def fgt_per_regime(d) -> "dict[int, float]":
    out = {}
    for j in range(len(d["boundaries"])):
        ts, accs = holdout_curve(d, j)
        if len(accs):
            out[j] = float(accs.max() - accs[-1])
    return out


def summarize(d) -> dict:
    s = dict(ADAPT=adapt(d), ADAPT_masked=adapt(d, masked=True), RET=ret(d))
    for j, v in ret_per_regime(d).items():
        s[f"RET_R{j}"] = v
    for k, b in enumerate(d["boundaries"]):
        s[f"POST_{k}"] = post_drift(d, int(b))
        s[f"POST_{k}_masked"] = post_drift(d, int(b), masked=True)
    return s
