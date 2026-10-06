"""
合成时序漂移数据集生成器

每个生成器返回一个 SyntheticDataset，包含：
  - X: (n_samples, n_features) 特征矩阵
  - y: (n_samples,) 二分类标签
  - regime_labels: (n_samples,) 真实体制标签（0-based int）
  - drift_points: List[int] 漂移发生的时间步索引
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class SyntheticDataset:
    X: np.ndarray
    y: np.ndarray
    regime_labels: np.ndarray
    drift_points: List[int]
    name: str


def make_rotating_boundary(
    n_samples: int = 10000,
    n_features: int = 2,
    drift_speed: float = 0.003,
    noise: float = 0.1,
    random_seed: int = 42,
) -> SyntheticDataset:
    """
    渐进关系漂移：决策边界（超平面法向量）随时间匀速旋转。

    特征从标准高斯分布采样。
    决策边界初始沿第一个特征轴（w = [1, 0, ..., 0]），
    每个时间步在 (f0, f1) 平面内旋转 drift_speed 弧度。

    类别 0/1 由样本落在边界哪侧决定，加入 noise 比例的标签翻转。
    """
    rng = np.random.default_rng(random_seed)
    X = rng.standard_normal((n_samples, n_features))

    y = np.empty(n_samples, dtype=int)
    regime_labels = np.zeros(n_samples, dtype=int)
    # 体制标签按旋转角度四等分，方便后续可视化
    angles = np.arange(n_samples) * drift_speed
    regime_labels = (angles / (np.pi / 2)).astype(int)  # 每 90° 换一个体制编号

    for t in range(n_samples):
        angle = t * drift_speed
        # 法向量在 (f0, f1) 平面内旋转
        w = np.zeros(n_features)
        w[0] = np.cos(angle)
        if n_features > 1:
            w[1] = np.sin(angle)
        label = int(np.dot(X[t], w) >= 0)
        # 标签噪声
        if rng.random() < noise:
            label = 1 - label
        y[t] = label

    # 人工标注几个"显著"漂移点（每 90° 旋转为一个自然断点）
    quarter = int(np.pi / 2 / drift_speed)
    drift_points = [quarter * i for i in range(1, int(n_samples // quarter))]

    return SyntheticDataset(
        X=X.astype(np.float32),
        y=y,
        regime_labels=regime_labels,
        drift_points=drift_points,
        name="rotating_boundary",
    )


def make_regime_switching(
    n_samples: int = 10000,
    n_features: int = 10,
    n_regimes: int = 3,
    regime_length: int = 2000,
    noise: float = 0.1,
    random_seed: int = 42,
) -> SyntheticDataset:
    """
    突变漂移：数据在多个统计体制之间突然切换。

    每个体制有独立的：
      - 特征均值向量 μ_k
      - 线性分类权重 w_k

    体制顺序：0 → 1 → 2 → ... → (n_regimes-1) → 0（循环），
    每个体制持续 regime_length 步。
    """
    rng = np.random.default_rng(random_seed)

    # 为每个体制生成独立参数
    means = rng.uniform(-2, 2, size=(n_regimes, n_features))
    weights = rng.standard_normal((n_regimes, n_features))
    weights /= np.linalg.norm(weights, axis=1, keepdims=True)  # 归一化

    X = np.empty((n_samples, n_features), dtype=np.float32)
    y = np.empty(n_samples, dtype=int)
    regime_labels = np.empty(n_samples, dtype=int)
    drift_points = []

    for t in range(n_samples):
        regime = (t // regime_length) % n_regimes
        regime_labels[t] = regime

        # 记录体制切换时刻
        if t > 0 and t % regime_length == 0:
            drift_points.append(t)

        x = rng.standard_normal(n_features) + means[regime]
        X[t] = x.astype(np.float32)

        # 居中决策边界：相对于体制均值判定，保证体制内类别平衡
        label = int(np.dot(x - means[regime], weights[regime]) >= 0)
        if rng.random() < noise:
            label = 1 - label
        y[t] = label

    return SyntheticDataset(
        X=X,
        y=y,
        regime_labels=regime_labels,
        drift_points=drift_points,
        name="regime_switching",
    )


def make_combined_drift(
    n_samples: int = 10000,
    n_features: int = 10,
    noise: float = 0.1,
    random_seed: int = 42,
) -> SyntheticDataset:
    """
    混合漂移：特征分布（均值）渐进漂移 + 每 3000 步突变决策边界。

    兼具两种漂移类型：
      - 特征语义漂移：特征均值随时间线性移动
      - 关系漂移：决策权重定期突变
    """
    rng = np.random.default_rng(random_seed)

    # 决策边界每 3000 步突变一次
    boundary_change_interval = 3000
    n_boundaries = n_samples // boundary_change_interval + 1
    weights_list = rng.standard_normal((n_boundaries, n_features))
    weights_list /= np.linalg.norm(weights_list, axis=1, keepdims=True)

    X = np.empty((n_samples, n_features), dtype=np.float32)
    y = np.empty(n_samples, dtype=int)
    regime_labels = np.empty(n_samples, dtype=int)
    drift_points = list(range(boundary_change_interval, n_samples, boundary_change_interval))

    # 均值漂移速度：前 5 个特征随时间线性偏移
    mean_drift_rate = 2.0 / n_samples  # 整个过程内漂移 2 个标准差

    for t in range(n_samples):
        boundary_idx = t // boundary_change_interval
        regime_labels[t] = boundary_idx

        # 特征均值渐进漂移
        mean_offset = np.zeros(n_features)
        mean_offset[:5] = t * mean_drift_rate

        x = rng.standard_normal(n_features) + mean_offset
        X[t] = x.astype(np.float32)

        # 居中决策边界：相对于当前均值偏移判定，保证类别平衡
        label = int(np.dot(x - mean_offset, weights_list[boundary_idx]) >= 0)
        if rng.random() < noise:
            label = 1 - label
        y[t] = label

    return SyntheticDataset(
        X=X,
        y=y,
        regime_labels=regime_labels,
        drift_points=drift_points,
        name="combined_drift",
    )


# 便捷工厂函数
_GENERATORS = {
    "rotating_boundary": make_rotating_boundary,
    "regime_switching": make_regime_switching,
    "combined_drift": make_combined_drift,
}


def make_dataset(name: str, **kwargs) -> SyntheticDataset:
    """按名称生成合成数据集。"""
    if name not in _GENERATORS:
        raise ValueError(f"未知数据集: {name}. 可选: {list(_GENERATORS.keys())}")
    return _GENERATORS[name](**kwargs)
