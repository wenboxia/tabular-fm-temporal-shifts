"""
时序窗口化加载器

将时序数据集拆分为一系列 (上下文窗口, 查询样本) 对，
供 TabPFN in-context learning 使用。

用法示例：
    dataset = make_rotating_boundary(n_samples=10000)
    loader = TemporalWindowLoader(dataset.X, dataset.y, context_size=500, step_size=1)
    for batch in loader:
        X_ctx, y_ctx, X_query, y_query, t = batch
        ...
"""

from dataclasses import dataclass
from typing import Generator, Optional

import numpy as np


@dataclass
class TemporalBatch:
    """单个时间步的数据批次。"""
    X_ctx: np.ndarray      # 上下文特征 (context_size, n_features)
    y_ctx: np.ndarray      # 上下文标签 (context_size,)
    X_query: np.ndarray    # 当前查询特征 (1, n_features)
    y_query: np.ndarray    # 当前真实标签 (1,)
    t: int                 # 时间步索引


class TemporalWindowLoader:
    """
    滑动窗口时序加载器。

    对于每个时间步 t（从 context_size 开始），
    返回：
      - 上下文窗口：[t - context_size, t) 的历史数据
      - 查询样本：时间步 t 的单个样本

    这是 prequential（先测试后训练）评估的标准设置：
    先用历史数据预测当前样本，再将当前样本的真实标签用于更新。
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        context_size: int = 500,
        step_size: int = 1,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ):
        """
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签向量 (n_samples,)
            context_size: 上下文窗口大小（TabPFN 的"训练集"样本数）
            step_size: 每次滑动的步长（通常为 1）
            start: 起始时间步（默认 = context_size，确保上下文非空）
            end: 结束时间步（默认 = n_samples）
        """
        assert len(X) == len(y), "X 和 y 长度必须一致"
        assert context_size >= 1, "context_size 至少为 1"

        self.X = X
        self.y = y
        self.context_size = context_size
        self.step_size = step_size
        self.start = start if start is not None else context_size
        self.end = end if end is not None else len(X)

        assert self.start >= context_size, (
            f"start ({self.start}) 必须 >= context_size ({context_size})"
        )

    def __len__(self) -> int:
        """返回总时间步数。"""
        return max(0, (self.end - self.start + self.step_size - 1) // self.step_size)

    def __iter__(self) -> Generator[TemporalBatch, None, None]:
        """迭代所有时间步，每次 yield 一个 TemporalBatch。"""
        for t in range(self.start, self.end, self.step_size):
            X_ctx = self.X[t - self.context_size: t]
            y_ctx = self.y[t - self.context_size: t]
            X_query = self.X[t: t + 1]
            y_query = self.y[t: t + 1]
            yield TemporalBatch(
                X_ctx=X_ctx,
                y_ctx=y_ctx,
                X_query=X_query,
                y_query=y_query,
                t=t,
            )

    def get_batch(self, t: int) -> TemporalBatch:
        """随机访问：获取时间步 t 的批次。"""
        assert self.context_size <= t < self.end, (
            f"t={t} 超出范围 [{self.context_size}, {self.end})"
        )
        return TemporalBatch(
            X_ctx=self.X[t - self.context_size: t],
            y_ctx=self.y[t - self.context_size: t],
            X_query=self.X[t: t + 1],
            y_query=self.y[t: t + 1],
            t=t,
        )


class CompositeWindowLoader:
    """
    组合窗口加载器：固定代表集 + 滑动近期窗。

    将 context_size 拆分为两部分：
      - fixed_pool (fixed_size 个样本)：从最早的 pool_source_size 个样本中
        随机采样，在整个流中保持不变。提供"长期记忆"。
      - sliding_window (context_size - fixed_size 个样本)：紧邻查询点的最近样本。
        提供"短期记忆"，快速适应漂移。

    当 fixed_ratio=0.0 时退化为纯滑动窗口（等价于 TemporalWindowLoader）。
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        context_size: int = 300,
        fixed_ratio: float = 0.67,
        pool_source_size: Optional[int] = None,
        step_size: int = 1,
        random_seed: int = 42,
    ):
        """
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签向量 (n_samples,)
            context_size: 总上下文大小（fixed + sliding）
            fixed_ratio: 固定池占 context_size 的比例，范围 [0, 1)
            pool_source_size: 从前多少个样本中采样固定池（默认 = context_size）
            step_size: 滑动步长
            random_seed: 固定池采样的随机种子
        """
        assert len(X) == len(y)
        assert 0.0 <= fixed_ratio < 1.0, "fixed_ratio 必须在 [0, 1) 范围内"

        self.X = X
        self.y = y
        self.context_size = context_size
        self.fixed_ratio = fixed_ratio
        self.step_size = step_size

        self.fixed_size = int(context_size * fixed_ratio)
        self.sliding_size = context_size - self.fixed_size

        # 固定池：从最早的 pool_source_size 个样本中采样
        if self.fixed_size > 0:
            source_size = pool_source_size if pool_source_size else context_size
            source_size = min(source_size, len(X))
            assert self.fixed_size <= source_size, (
                f"fixed_size ({self.fixed_size}) > pool_source_size ({source_size})"
            )
            rng = np.random.default_rng(random_seed)
            indices = rng.choice(source_size, size=self.fixed_size, replace=False)
            indices.sort()  # 保持时间顺序
            self.fixed_X = X[indices].copy()
            self.fixed_y = y[indices].copy()
        else:
            self.fixed_X = np.empty((0, X.shape[1]), dtype=X.dtype)
            self.fixed_y = np.empty((0,), dtype=y.dtype)

        # 滑动窗口的起始位置：需要至少 sliding_size 个历史样本
        self.start = max(context_size, self.sliding_size)
        self.end = len(X)

    def __len__(self) -> int:
        return max(0, (self.end - self.start + self.step_size - 1) // self.step_size)

    def __iter__(self) -> Generator[TemporalBatch, None, None]:
        for t in range(self.start, self.end, self.step_size):
            # 滑动部分：紧邻查询点的最近样本
            slide_start = t - self.sliding_size
            X_slide = self.X[slide_start: t]
            y_slide = self.y[slide_start: t]

            # 拼接：固定池 + 滑动窗口
            X_ctx = np.concatenate([self.fixed_X, X_slide], axis=0)
            y_ctx = np.concatenate([self.fixed_y, y_slide], axis=0)

            yield TemporalBatch(
                X_ctx=X_ctx,
                y_ctx=y_ctx,
                X_query=self.X[t: t + 1],
                y_query=self.y[t: t + 1],
                t=t,
            )


def evict_oldest_of_majority(long_idx: "list[int]", y: np.ndarray) -> None:
    """淘汰"当前数量最多的那一类里最老的一条"（KDD 2026 双记忆的长期库规则）。

    long_idx 按时间升序；数量并列时取类别编号较小的那一类。
    `DualMemoryLoader` 与 round 2 的 `DualMemory` 共用这个函数，保证两者行为一致。
    """
    counts: dict = {}
    for i in long_idx:
        counts[int(y[i])] = counts.get(int(y[i]), 0) + 1
    majority = max(counts, key=lambda c: (counts[c], -c))
    for pos, i in enumerate(long_idx):
        if int(y[i]) == majority:
            long_idx.pop(pos)
            return
    long_idx.pop(0)


class DualMemoryLoader:
    """长短双记忆 context（Phase 5.5 Step 7）。

    参照 KDD 2026 (Lourenço & Gama, *In-context Learning of Evolving Data Streams
    with Tabular Foundational Models*) 的双记忆方案，作为 Phase 1 纯滑窗之外的
    **文献基线**，避免只和弱基线比较。

    机制：
      - **短期库**：容量 `int(budget * short_ratio)` 的 FIFO，装最近样本；
      - **长期库**：短期库溢出的样本流入长期库；长期库满时，淘汰**当前数量最多的
        那一类里最老的**一条（类均衡保留）。
      - context = 长期库 + 短期库（TabPFN 对 context 顺序不敏感）。

    ⚠️ 两个已知陷阱，都在这里处理掉了：

    1. **朴素实现会退化成纯滑窗**。长期库一旦填满，就变成"每步进一条、出一条"，
       在类别均衡的流上进出速率相同，长短两库的并集 ≈ 最近 budget 条，
       与 `TemporalWindowLoader` 的 context 是同一个集合。
       所以本实现的淘汰规则是**按类**挑最老的，只有当某类在长期库里过量时才淘汰它，
       在类别不均衡的时段（Insects 的单类长段）才真正与滑窗不同。
    2. **陈旧样本会被永久钉住**。少数类的样本永远不是"最多的那一类"，
       于是可以无限期留在长期库里，把过时的 P(y|x) 一直喂给 TabPFN。
       `max_age` 给长期库的样本设年龄上限，超龄一律淘汰。

    Prequential 安全：先 yield 再 push（yield-then-push），
    所以 (X[t], y[t]) 绝不会出现在预测它自己时的 context 里。
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        context_size: int = 300,
        short_ratio: float = 0.5,
        max_age: Optional[int] = None,
        step_size: int = 1,
        warmup: Optional[int] = None,
    ):
        """
        Args:
            X, y:          数据
            context_size:  长短两库容量之和（与纯滑窗的 context_size 对齐，保证公平）
            short_ratio:   短期库占比 ∈ (0, 1)
            max_age:       长期库样本的最大年龄（步）；None = 不限（不推荐，见类注释陷阱 2）
            step_size:     步长
            warmup:        前多少步只用滑窗热身（默认 = context_size）
        """
        assert len(X) == len(y)
        assert 0.0 < short_ratio < 1.0, f"short_ratio 必须 ∈ (0,1)，收到 {short_ratio}"
        assert context_size >= 2, context_size
        assert max_age is None or max_age > 0, max_age

        self.X = X
        self.y = y
        self.context_size = context_size
        self.short_ratio = short_ratio
        self.max_age = max_age
        self.step_size = step_size

        self.short_capacity = max(1, int(round(context_size * short_ratio)))
        self.long_capacity = context_size - self.short_capacity
        assert self.long_capacity >= 1, (
            f"long_capacity={self.long_capacity} < 1；short_ratio 太大"
        )

        self.start = warmup if warmup is not None else context_size
        self.end = len(X)

    def __len__(self) -> int:
        return max(0, (self.end - self.start + self.step_size - 1) // self.step_size)

    # ------------------------------------------------------------------

    def _evict_from_long(self, long_idx: "list[int]", t: int) -> None:
        """长期库满时淘汰一条：先清超龄，再淘汰"最多类里最老的那条"。"""
        if self.max_age is not None:
            fresh = [i for i in long_idx if t - i <= self.max_age]
            if len(fresh) < len(long_idx):
                long_idx[:] = fresh
                if len(long_idx) < self.long_capacity:
                    return
        evict_oldest_of_majority(long_idx, self.y)

    def __iter__(self) -> Generator[TemporalBatch, None, None]:
        long_idx: list = []
        short_idx: list = []

        # 热身：用 [0, start) 填充两库（短期库拿最近的，其余进长期库）
        for i in range(self.start):
            short_idx.append(i)
            if len(short_idx) > self.short_capacity:
                overflow = short_idx.pop(0)
                long_idx.append(overflow)
                if len(long_idx) > self.long_capacity:
                    self._evict_from_long(long_idx, self.start)

        for t in range(self.start, self.end, self.step_size):
            ctx = long_idx + short_idx                     # 长在前、短在后
            idx = np.asarray(ctx, dtype=np.int64)
            yield TemporalBatch(
                X_ctx=self.X[idx],
                y_ctx=self.y[idx],
                X_query=self.X[t: t + 1],
                y_query=self.y[t: t + 1],
                t=t,
            )
            # yield-then-push：标签观测之后才入库，绝不泄漏当前样本
            short_idx.append(t)
            if len(short_idx) > self.short_capacity:
                overflow = short_idx.pop(0)
                long_idx.append(overflow)
                if len(long_idx) > self.long_capacity:
                    self._evict_from_long(long_idx, t)
