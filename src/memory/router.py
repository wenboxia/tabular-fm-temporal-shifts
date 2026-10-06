"""Round 2 路由器：查询更像哪个已存档的旧阶段？

规则在 pilot 之前固定（`results/phase56_prereg.md`）：
  1. pool = 当前滑窗 W 的行 + 所有档案的行，按 pool 自身的均值/标准差 z-score；
  2. 对每个查询取 k 个最近邻（欧氏距离，float64，距离相同按 pool 内位置先后）；
  3. 每个来源的得分 = 近邻里来自它的个数 / 它的行数（近似密度比）；
  4. 只有当得分最高的档案**严格高于** W 的得分时才路由到该档案，否则只用 W。

路由只看特征、不看标签；pool 里只有已经揭示过的流内行，不含任何留出行。
"""
from __future__ import annotations

import numpy as np

WINDOW = -1


def route_queries(
    X_query: np.ndarray,
    X_window: np.ndarray,
    X_archives: "list[np.ndarray]",
    k: int = 10,
) -> np.ndarray:
    """返回每个查询的路由：WINDOW (-1) 或档案编号 0..m-1。"""
    n_q = len(X_query)
    m = len(X_archives)
    if n_q == 0:
        return np.zeros(0, dtype=np.int64)
    if m == 0 or len(X_window) == 0:
        return np.full(n_q, WINDOW, dtype=np.int64)

    parts = [np.asarray(X_window, dtype=np.float64)] + [np.asarray(a, dtype=np.float64) for a in X_archives]
    src = np.concatenate([np.full(len(p), i - 1, dtype=np.int64) for i, p in enumerate(parts)])
    pool = np.concatenate(parts, axis=0)
    mu = pool.mean(axis=0)
    sd = pool.std(axis=0)
    sd[sd < 1e-6] = 1.0
    P = (pool - mu) / sd
    Q = (np.asarray(X_query, dtype=np.float64) - mu) / sd

    kk = min(k, len(P))
    sizes = np.array([len(p) for p in parts], dtype=np.float64)   # [W, A0, A1, ...]
    out = np.empty(n_q, dtype=np.int64)
    for s in range(0, n_q, 512):
        q = Q[s: s + 512]
        d2 = (q * q).sum(1)[:, None] + (P * P).sum(1)[None, :] - 2.0 * q @ P.T
        nn = np.argsort(d2, axis=1, kind="stable")[:, :kk]      # 并列时 pool 位置靠前者优先
        votes = np.zeros((len(q), m + 1))
        for j in range(m + 1):
            votes[:, j] = (src[nn] == j - 1).sum(1)
        score = votes / sizes[None, :]
        best_arch = np.argmax(score[:, 1:], axis=1)            # 并列取编号小的档案
        best_score = score[np.arange(len(q)), best_arch + 1]
        out[s: s + len(q)] = np.where(best_score > score[:, 0], best_arch, WINDOW)
    return out
