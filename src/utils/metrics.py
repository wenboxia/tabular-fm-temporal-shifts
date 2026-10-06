"""
评估指标

核心指标：
  - prequential_accuracy: 逐步预测准确率（先预测后更新）
  - window_accuracy: 按时间窗口计算的准确率序列
  - adaptation_speed: 漂移后恢复到基线准确率所需步数
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def prequential_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    fading_factor: float = 1.0,
) -> float:
    """
    Prequential（先测试后训练）准确率。

    fading_factor < 1 时，对更近的预测赋予更高权重（衰减加权）。
    fading_factor = 1 等价于普通准确率。

    Args:
        predictions: (n,) 预测标签
        labels:      (n,) 真实标签
        fading_factor: 衰减因子，范围 (0, 1]

    Returns:
        加权准确率，范围 [0, 1]
    """
    n = len(predictions)
    assert len(labels) == n
    if n == 0:
        return 0.0

    correct = (predictions == labels).astype(float)
    if fading_factor == 1.0:
        return float(np.mean(correct))

    # 指数衰减权重：最近的样本权重最高
    weights = np.array([fading_factor ** (n - 1 - i) for i in range(n)])
    return float(np.average(correct, weights=weights))


def window_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    window_size: int = 200,
) -> np.ndarray:
    """
    按滑动窗口计算准确率序列。

    用于可视化模型性能随时间的变化，特别是在漂移点附近的跌落和恢复。

    Args:
        predictions: (n,) 预测标签
        labels:      (n,) 真实标签
        window_size: 窗口大小

    Returns:
        accs: (n - window_size + 1,) 每个窗口的准确率
    """
    n = len(predictions)
    assert len(labels) == n
    correct = (predictions == labels).astype(float)

    # 用累积和实现 O(n) 滑动窗口
    cumsum = np.concatenate([[0], np.cumsum(correct)])
    window_sums = cumsum[window_size:] - cumsum[:-window_size]
    return window_sums / window_size


def adaptation_speed(
    predictions: np.ndarray,
    labels: np.ndarray,
    drift_points: List[int],
    baseline_acc: float,
    window_size: int = 50,
    offset: int = 0,
) -> Dict[int, Optional[int]]:
    """
    计算每个漂移点后恢复到 baseline_acc 所需的步数。

    Args:
        predictions:  (n,) 预测标签
        labels:       (n,) 真实标签
        drift_points: 漂移发生的时间步列表（全局索引）
        baseline_acc: 参考准确率（通常是漂移前的平均值）
        window_size:  用于计算局部准确率的小窗口
        offset:       predictions 数组对应的起始全局时间步

    Returns:
        dict: {drift_point -> 恢复步数 (None 表示未恢复)}
    """
    result = {}
    n = len(predictions)
    correct = (predictions == labels).astype(float)

    for dp in drift_points:
        local_start = dp - offset
        if local_start < 0 or local_start >= n:
            result[dp] = None
            continue

        recovered_steps = None
        for i in range(local_start, min(n - window_size, n)):
            window_acc = np.mean(correct[i: i + window_size])
            if window_acc >= baseline_acc:
                recovered_steps = i - local_start
                break
        result[dp] = recovered_steps

    return result


def summarize_results(
    predictions: np.ndarray,
    labels: np.ndarray,
    drift_points: List[int],
    window_size: int = 200,
    offset: int = 0,
) -> Dict:
    """
    一次性计算并返回所有关键指标的汇总字典。

    Args:
        predictions:  (n,) 预测标签
        labels:       (n,) 真实标签
        drift_points: 漂移点列表（全局索引）
        window_size:  窗口大小
        offset:       predictions 起始的全局时间步

    Returns:
        包含 overall_acc, pre_drift_acc, post_drift_acc, window_accs,
        avg_adaptation_speed 的字典
    """
    overall = prequential_accuracy(predictions, labels)
    win_accs = window_accuracy(predictions, labels, window_size)

    # 漂移前后准确率（各取最近 window_size 步作为代表）
    pre_drift_accs, post_drift_accs = [], []
    for dp in drift_points:
        local = dp - offset
        if 0 < local < len(predictions):
            pre_start = max(0, local - window_size)
            pre_drift_accs.append(float(np.mean(predictions[pre_start:local] == labels[pre_start:local])))
            post_end = min(len(predictions), local + window_size)
            post_drift_accs.append(float(np.mean(predictions[local:post_end] == labels[local:post_end])))

    baseline = float(np.mean(pre_drift_accs)) if pre_drift_accs else overall
    speeds = adaptation_speed(predictions, labels, drift_points, baseline_acc=baseline * 0.95, offset=offset)
    valid_speeds = [v for v in speeds.values() if v is not None]

    # Balanced accuracy 和 AUC-ROC
    bal_acc = float(balanced_accuracy_score(labels, predictions))
    try:
        auc = float(roc_auc_score(labels, predictions))
    except ValueError:
        auc = None  # 只有一个类别时 AUC 无定义

    return {
        "overall_acc": overall,
        "balanced_acc": bal_acc,
        "auc_roc": auc,
        "pre_drift_acc": float(np.mean(pre_drift_accs)) if pre_drift_accs else None,
        "post_drift_acc": float(np.mean(post_drift_accs)) if post_drift_accs else None,
        "window_accs": win_accs,
        "avg_adaptation_speed": float(np.mean(valid_speeds)) if valid_speeds else None,
    }


# ---------------------------------------------------------------------------
# Phase 5.5：共用恢复目标 + 少犯错误数
# ---------------------------------------------------------------------------
#
# 为什么要新指标（既有 `adaptation_speed` 的两个问题）：
#
# 1. **门槛是各方法自己的**。`summarize_results` 用该方法**自己**漂移前的均值 × 0.95
#    作为恢复目标，所以一个整体更差的方法门槛更低、反而显得"恢复更快"。
#    跨方法比较必须用**共用**目标 A*（通常取同段同 seed 的 Phase 1 基线）。
# 2. **恢复时刻记在窗口起点**。原实现一旦发现某个 [i, i+w) 窗口达标就返回 i，
#    而该窗口要到 i+w-1 才走完，最多提前 w-1 步。
#    `recovery_step(..., confirm_at_window_end=True)` 记在窗口末尾。
#
# 另外提供 `errors_avoided`：固定窗口内比参考方法少犯了几次错。
# 它不依赖任何阈值，可以分别从**变点**和从**报警时刻**起算 ——
# 后者正是"报警之后的动作到底值不值"的直接答案。


def recovery_step(
    predictions: np.ndarray,
    labels: np.ndarray,
    drift_point: int,
    target_acc: float,
    window_size: int = 100,
    offset: int = 0,
    require_consecutive: int = 2,
    confirm_at_window_end: bool = True,
) -> Optional[int]:
    """漂移后恢复到**共用**目标 target_acc 所需步数；未恢复返回 None。

    Args:
        target_acc:            共用目标 A*（跨方法同一个值，不用各自的漂移前均值）
        require_consecutive:   需要连续多少个**不重叠**窗口达标才算恢复（防抖动误判）
        confirm_at_window_end: True = 恢复时刻记在确认窗口的末尾（默认，诚实口径）

    未恢复返回 None，**不要**在求均值时丢弃它们 —— 那会让差方法看起来更快。
    """
    n = len(predictions)
    local = drift_point - offset
    if local < 0 or local >= n:
        return None
    correct = (predictions == labels).astype(float)

    streak = 0
    i = local
    while i + window_size <= n:
        if float(np.mean(correct[i: i + window_size])) >= target_acc:
            streak += 1
            if streak >= require_consecutive:
                end = i + window_size
                return int((end if confirm_at_window_end else i + window_size - 1) - local)
        else:
            streak = 0
        i += window_size          # 不重叠窗口
    return None


def errors_avoided(
    predictions: np.ndarray,
    labels: np.ndarray,
    reference_predictions: np.ndarray,
    from_t: int,
    horizon: int = 300,
    offset: int = 0,
) -> Optional[Dict]:
    """从 from_t 起 horizon 步内，本方法比参考方法**少犯**了几次错。

    正值 = 更好。不依赖任何阈值，因此避开了恢复目标的任意性。
    from_t 传变点 → 衡量整体漂移应对；传报警时刻 → 直接衡量"动作值不值"。
    """
    n = len(predictions)
    if len(reference_predictions) != n or len(labels) != n:
        raise ValueError("predictions / reference / labels 长度必须一致")
    lo = from_t - offset
    if lo < 0 or lo >= n:
        return None
    hi = min(n, lo + horizon)
    ours = int(np.sum(predictions[lo:hi] != labels[lo:hi]))
    ref = int(np.sum(reference_predictions[lo:hi] != labels[lo:hi]))
    return {
        "from_t": int(from_t), "n_steps": int(hi - lo),
        "errors_ours": ours, "errors_ref": ref,
        "errors_avoided": ref - ours,
        "acc_delta_pp": 100.0 * (ref - ours) / max(1, hi - lo),
    }


def shared_target_summary(
    predictions: np.ndarray,
    labels: np.ndarray,
    drift_points: List[int],
    target_acc: float,
    reference_predictions: Optional[np.ndarray] = None,
    alarm_times: Optional[List[int]] = None,
    window_size: int = 100,
    horizon: int = 300,
    offset: int = 0,
) -> Dict:
    """Phase 5.5 的跨方法可比汇总。

    Returns 里 `n_not_recovered` 必须和 `avg_recovery_step` 一起读：
    平均值只统计恢复了的漂移，未恢复的单独计数，不能被静默丢掉。
    """
    recs = {
        int(dp): recovery_step(predictions, labels, dp, target_acc,
                               window_size=window_size, offset=offset)
        for dp in drift_points
    }
    ok = [v for v in recs.values() if v is not None]

    out: Dict = {
        "target_acc": float(target_acc),
        "recovery_steps": recs,
        "avg_recovery_step": float(np.mean(ok)) if ok else None,
        "n_not_recovered": int(sum(v is None for v in recs.values())),
    }
    if reference_predictions is not None:
        out["errors_avoided_from_drift"] = [
            errors_avoided(predictions, labels, reference_predictions, dp,
                           horizon=horizon, offset=offset)
            for dp in drift_points
        ]
        if alarm_times:
            out["errors_avoided_from_alarm"] = [
                errors_avoided(predictions, labels, reference_predictions, a,
                               horizon=horizon, offset=offset)
                for a in alarm_times
            ]
    return out
