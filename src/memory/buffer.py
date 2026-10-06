"""
FIFO 工作记忆缓冲区

存储最近若干步的 (特征, 预测误差) 对，
供快速校正器查询使用。

支持两种校正策略：
  - KNN：找最相似的 K 个历史样本，取其误差均值作为补偿
  - EMA：对缓冲区内所有误差做指数移动平均
"""

from collections import deque
from typing import Optional

import numpy as np


class WorkingMemoryBuffer:
    """
    固定容量的 FIFO 工作记忆缓冲区。

    每个 entry 存储：
      - feature:  (n_features,) 该时步的原始特征
      - error:    float，该时步的预测误差 = y_true - y_pred_slow（标量）
    """

    def __init__(self, capacity: int = 100):
        """
        Args:
            capacity: 缓冲区最大容量，超出后自动丢弃最旧的 entry
        """
        assert capacity >= 1
        self.capacity = capacity
        self._features: deque = deque(maxlen=capacity)
        self._errors: deque = deque(maxlen=capacity)

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def push(self, feature: np.ndarray, error: float) -> None:
        """将一个新观测压入缓冲区（旧 entry 自动淘汰）。"""
        self._features.append(np.asarray(feature, dtype=np.float32).copy())
        self._errors.append(float(error))

    def clear(self) -> None:
        """清空缓冲区（巩固后调用）。"""
        self._features.clear()
        self._errors.clear()

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._errors)

    def is_empty(self) -> bool:
        return len(self._errors) == 0

    def knn_correction(self, query: np.ndarray, k: int = 5) -> float:
        """
        K 近邻校正：在缓冲区中找最相似的 K 个历史特征，
        返回它们的误差均值作为当前时步的补偿量。

        Args:
            query: (n_features,) 当前查询特征
            k:     近邻数量

        Returns:
            校正量（标量），加到 y_slow 上
        """
        if self.is_empty():
            return 0.0

        features = np.stack(self._features)          # (n, d)
        errors = np.array(self._errors)              # (n,)
        q = np.asarray(query, dtype=np.float32).ravel()

        # 欧氏距离
        diffs = features - q[np.newaxis, :]
        dists = np.sum(diffs ** 2, axis=1)           # 不开根号，比较排序即可

        k = min(k, len(errors))
        topk_idx = np.argpartition(dists, k - 1)[:k]
        return float(np.mean(errors[topk_idx]))

    def ema_correction(self, alpha: float = 0.1) -> float:
        """
        指数移动平均校正：对缓冲区内所有误差做指数衰减加权，
        最近的误差权重最高。

        Args:
            alpha: EMA 平滑系数，越大越偏向最近误差

        Returns:
            加权平均误差（标量）
        """
        if self.is_empty():
            return 0.0

        errors = np.array(self._errors)   # 时序顺序，最旧在前
        n = len(errors)
        # 权重：最新的样本（index n-1）权重最高
        weights = np.array([alpha * (1 - alpha) ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()
        return float(np.dot(weights, errors))

    # ------------------------------------------------------------------
    # 状态查询（供巩固机制使用）
    # ------------------------------------------------------------------

    def recent_errors(self, window: int) -> np.ndarray:
        """返回最近 window 个误差（若不足则返回全部）。"""
        errors = np.array(self._errors)
        return errors[-window:] if len(errors) >= window else errors

    def recent_features(self, window: int) -> np.ndarray:
        """
        返回最近 window 个时步的特征矩阵（若不足则返回全部）。

        Args:
            window: 取最近多少步的特征

        Returns:
            (min(window, len), n_features) float32 数组
        """
        features = np.array(list(self._features), dtype=np.float32)
        return features[-window:] if len(features) >= window else features

    def mean_recent_error(self, window: int = 50) -> float:
        """最近 window 步的平均误差。"""
        recent = self.recent_errors(window)
        return float(np.mean(recent)) if len(recent) > 0 else 0.0

    def std_recent_error(self, window: int = 50) -> float:
        """最近 window 步的误差标准差。"""
        recent = self.recent_errors(window)
        return float(np.std(recent)) if len(recent) > 1 else 0.0
